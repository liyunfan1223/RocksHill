//
// Created by Yunfan Li on 2023/5/23.
//

#pragma once

#include <rocksdb/status.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache/sharded_cache.h"
#include "port/malloc.h"
#include "rocksdb/advanced_cache.h"
#include "rocksdb/slice.h"
#include "util/autovector.h"
#include "util/distributed_mutex.h"
namespace ROCKSDB_NAMESPACE {

namespace hill {

namespace {

const double EPSILON = 1e-10;

enum class RC {
  DEFAULT,
  SUCCESS,
  HIT,
  MISS,
  FAILED,
  UNIMPLEMENT,
};

}  // namespace
class HillCache;
struct HillHandle {
  Cache::ObjectPtr value;
  const Cache::CacheItemHelper* helper;
  size_t total_charge;  // TODO(opt): Only allow uint32_t?
  size_t key_length;
  uint32_t hash;
  // The number of external refs to this entry. The cache itself is not counted.
  uint32_t refs;

  uint8_t m_flags;
  enum MFlags : uint8_t {
    // Whether this entry is referenced by the hash table.
    M_IN_CACHE = (1 << 0),
    // Whether this entry has had any lookups (hits).
    M_HAS_HIT = (1 << 1),
    // Whether this entry is in high-pri pool.
    M_IN_HIGH_PRI_POOL = (1 << 2),
    // Whether this entry is in low-pri pool.
    M_IN_LOW_PRI_POOL = (1 << 3),
  };

  // "Immutable" flags - only set in single-threaded context and then
  // can be accessed without mutex
  uint8_t im_flags;
  enum ImFlags : uint8_t {
    // Whether this entry is high priority entry.
    IM_IS_HIGH_PRI = (1 << 0),
    // Whether this entry is low priority entry.
    IM_IS_LOW_PRI = (1 << 1),
    // Marks result handles that should not be inserted into cache
    IM_IS_STANDALONE = (1 << 2),
  };

  char key_data[1];

  Slice key() const { return Slice(key_data, key_length); }

  // For HandleImpl concept
  uint32_t GetHash() const { return hash; }

  // Increase the reference count by 1.
  void Ref() { refs++; }

  // Just reduce the reference count by 1. Return true if it was last reference.
  bool Unref() {
    //// assert(refs > 0);
    refs--;
    return refs == 0;
  }

  // Return true if there are external refs, false otherwise.
  bool HasRefs() const { return refs > 0; }

  bool InCache() const { return m_flags & M_IN_CACHE; }
  bool IsHighPri() const { return im_flags & IM_IS_HIGH_PRI; }
  bool InHighPriPool() const { return m_flags & M_IN_HIGH_PRI_POOL; }
  bool IsLowPri() const { return im_flags & IM_IS_LOW_PRI; }
  bool InLowPriPool() const { return m_flags & M_IN_LOW_PRI_POOL; }
  bool HasHit() const { return m_flags & M_HAS_HIT; }
  bool IsStandalone() const { return im_flags & IM_IS_STANDALONE; }

  void SetInCache(bool in_cache) {
    if (in_cache) {
      m_flags |= M_IN_CACHE;
    } else {
      m_flags &= ~M_IN_CACHE;
    }
  }

  void SetPriority(Cache::Priority priority) {
    if (priority == Cache::Priority::HIGH) {
      im_flags |= IM_IS_HIGH_PRI;
      im_flags &= ~IM_IS_LOW_PRI;
    } else if (priority == Cache::Priority::LOW) {
      im_flags &= ~IM_IS_HIGH_PRI;
      im_flags |= IM_IS_LOW_PRI;
    } else {
      im_flags &= ~IM_IS_HIGH_PRI;
      im_flags &= ~IM_IS_LOW_PRI;
    }
  }

  void SetInHighPriPool(bool in_high_pri_pool) {
    if (in_high_pri_pool) {
      m_flags |= M_IN_HIGH_PRI_POOL;
    } else {
      m_flags &= ~M_IN_HIGH_PRI_POOL;
    }
  }

  void SetInLowPriPool(bool in_low_pri_pool) {
    if (in_low_pri_pool) {
      m_flags |= M_IN_LOW_PRI_POOL;
    } else {
      m_flags &= ~M_IN_LOW_PRI_POOL;
    }
  }

  void SetHit() { m_flags |= M_HAS_HIT; }

  void SetIsStandalone(bool is_standalone) {
    if (is_standalone) {
      im_flags |= IM_IS_STANDALONE;
    } else {
      im_flags &= ~IM_IS_STANDALONE;
    }
  }

  void Free(MemoryAllocator* allocator) {
    //// assert(refs == 0);
    //// assert(helper);
    if (helper->del_cb) {
      helper->del_cb(value, allocator);
    }

    free(this);
  }

  inline size_t CalcuMetaCharge(
      CacheMetadataChargePolicy metadata_charge_policy) const {
    if (metadata_charge_policy != kFullChargeCacheMetadata) {
      return 0;
    } else {
#ifdef ROCKSDB_MALLOC_USABLE_SIZE
      return malloc_usable_size(
          const_cast<void*>(static_cast<const void*>(this)));
#else
      // This is the size that is used when a new handle is created.
      return sizeof(LRUHandle) - 1 + key_length;
#endif
    }
  }

  // Calculate the memory usage by metadata.
  inline void CalcTotalCharge(
      size_t charge, CacheMetadataChargePolicy metadata_charge_policy) {
    total_charge = charge + CalcuMetaCharge(metadata_charge_policy);
  }

  inline size_t GetCharge(
      CacheMetadataChargePolicy metadata_charge_policy) const {
    size_t meta_charge = CalcuMetaCharge(metadata_charge_policy);
    //// assert(total_charge >= meta_charge);
    return total_charge - meta_charge;
  }
};

class Replacer {
 public:
  explicit Replacer(int32_t buffer_size, int32_t _stats_interval = -1)
      : buffer_size_(buffer_size), stats_interval(_stats_interval) {
    if (stats_interval != -1) {
      enable_interval_stats_ = true;
    }
  }

  virtual ~Replacer() = default;

  std::string stats() {
    std::stringstream s;
    s << get_name() << ":"
      << " buffer_size:" << buffer_size_ << " hit_count:" << hit_count_
      << " miss_count:" << miss_count_ << " hit_rate:"
      << (float)hit_count_ / (float)(hit_count_ + miss_count_) * 100 << "\%"
      << std::endl;
    return s.str();
  }

  virtual RC access(const std::string& key, HillHandle* h) = 0;
  virtual std::string get_name() = 0;
  virtual std::string get_configuration() { return {""}; }
  virtual RC check_consistency() { return RC::DEFAULT; }
  int32_t hit_count() const { return hit_count_; }
  int32_t miss_count() const { return miss_count_; }
  int32_t increase_hit_count() {
    hit_count_ += 1;
    return hit_count_;
  }
  int32_t increase_miss_count() {
    miss_count_ += 1;
    return miss_count_;
  }

 protected:
  const int32_t buffer_size_;
  int32_t hit_count_{};
  int32_t miss_count_{};
  bool enable_interval_stats_{false};
  int32_t stats_interval{};
  int ts{};

 private:
  friend class HillCache;
};

class HillSubReplacer {
  struct HillEntry {
    HillEntry() = default;
    HillEntry& operator=(const HillEntry& other) {
      if (this != &other) {
        key_iter = other.key_iter;
        insert_level = other.insert_level;
        insert_ts = other.insert_ts;
        h_recency = other.h_recency;
        h_ = other.h_;
      }
      return *this;
    }
    HillEntry(std::list<std::string>::iterator _key_iter, int _insert_level,
              int _insert_ts, bool _h_recency, HillHandle* h) {
      this->key_iter = _key_iter;
      this->insert_level = _insert_level;
      this->insert_ts = _insert_ts;
      this->h_recency = _h_recency;
      this->h_ = h;
    }

    std::list<std::string>::iterator key_iter;
    int insert_level{};
    int insert_ts{};
    bool h_recency{};  // 高时近性 说明在顶层LRU中
    HillHandle* h_;
  };

 public:
  HillSubReplacer(int32_t size, double init_half, double hit_points,
                  int max_points_bits, double ghost_size_ratio,
                  double top_ratio, int32_t _mru_threshold)
      : size_(size / 4096 + 1),
        init_half_(init_half),
        hit_points_(hit_points),
        max_points_bits_(max_points_bits),
        // ghost_size_ratio_(ghost_size_ratio),
        top_ratio_(top_ratio),
        mru_threshold_(_mru_threshold) {
    cur_half_ = init_half_;
    max_points_ = (1 << max_points_bits_) - 1;
    ghost_size_ = size_ * ghost_size_ratio;
    min_level_non_empty_ = max_points_;
    real_lru_.resize(max_points_);
    UpdateHalf(init_half_);
    lru_size_ = std::max(1, (int)(top_ratio_ * size_));
    ml_size_ = size_ - lru_size_;
  }

  int Access(const std::string& key, HillHandle* h) {
    bool hit = true;
    if (h) {
      //// assert(h->InCache());
      //// assert(key == h->key().ToString());
    }
    int32_t inserted_level = hit_points_;
    if (real_map_.count(key) == 0) {
      // miss
      hit = false;
      interval_miss_count_++;
      if (real_map_.size() == size_) {
        // Evict();
        // should never evict by replacer
        // if (h) {
        //// assert(false);
        // }
      }
      if (ghost_map_.count(key) != 0) {
        // use level in ghost
        std::list<std::string>::iterator hit_iter = ghost_map_[key].key_iter;
        int level = GetCurrentLevel(ghost_map_[key]);
        // erase key in ghost
        ghost_lru_.erase(hit_iter);
        ghost_map_.erase(key);
        inserted_level += level;
      }
    } else {
      // hit
      interval_hit_count_++;
      if (!real_map_[key].h_recency) {
        h1++;
        std::list<std::string>::iterator hit_iter = real_map_[key].key_iter;
        int level = GetCurrentLevel(real_map_[key]);  // real_map_[key].second;
        // erase key in real, use level in real
        real_lru_[level].erase(hit_iter);
        real_map_.erase(key);
        inserted_level += level;
        while (real_lru_[min_level_non_empty_].empty() &&
               min_level_non_empty_ < max_points_) {
          min_level_non_empty_++;
        }
        // //// assert(min_level_non_empty_ < 10);
      } else {
        h2++;
        inserted_level += real_map_[key].insert_level;
        top_lru_.erase(real_map_[key].key_iter);
        real_map_.erase(key);
      }
    }
    inserted_level = std::min(inserted_level, max_points_ - 1);

    if (top_lru_.size() >= lru_size_) {
      std::string& _key = top_lru_.back();
      HillHandle* oldH = real_map_[_key].h_;
      int lvl = real_map_[_key].insert_level;
      real_lru_[lvl].push_front(_key);
      // NOTE: 这里应该是把key塞进去。
      real_map_[_key] =
          HillEntry(real_lru_[lvl].begin(), lvl, cur_ts_, false, oldH);
      if (lvl < min_level_non_empty_) {
        min_level_non_empty_ = lvl;
      }
      top_lru_.pop_back();
    }
    top_lru_.push_front(key);
    // NOTE: 这里应该是把key塞进去。
    // if (h->value == nullptr) {
    //     std::cout << "Null value!, Key: " << Slice(key).ToString(true) <<
    //     "\n";
    // }
    real_map_[key] =
        HillEntry(top_lru_.begin(), inserted_level, cur_ts_, true, h);
    // auto entry = real_map_.find(key);
    // if(entry != real_map_.end()) {
    //     std::cout << "Insert: \nKey in: " << Slice(key).ToString(true) <<
    //     "\n"
    //               << "Key out: " << Slice(entry->first).ToString(true)
    //               << ", Handle address: " <<
    //               reinterpret_cast<std::uintptr_t>(entry->second.h_)
    //               << ", Key in handle: " << Slice(entry->second.h_->key_data,
    //               entry->second.h_->key_length).ToString(true)
    //               << ", Value address: " <<
    //               reinterpret_cast<std::uintptr_t>(entry->second.h_->value)
    //               << "\n";
    // }
    cur_ts_++;
    if (cur_ts_ >= next_rolling_ts_) {
      Rolling();
    }
    return hit;
  }

  HillHandle* Get(const std::string& key) {
    auto entry = real_map_.find(key);
    // std::cout << "Get: " << Slice(key).ToString(true) << '\n';
    // std::cout << "Get: " << Slice(key).ToString(true) << "\n";
    //           << "Key out: " << Slice(entry->first).ToString(true)
    //           << ", Handle address: " <<
    //           reinterpret_cast<std::uintptr_t>(entry->second.h_)
    //           << ", Key in handle: " << Slice(entry->second.h_->key_data,
    //           entry->second.h_->key_length).ToString(true)
    //           << ", Value address: " <<
    //           reinterpret_cast<std::uintptr_t>(entry->second.h_->value) <<
    //           "\n";
    if (entry != real_map_.end()) {
      //// assert(entry->second.h_->InCache());
      // return nullptr;
      return entry->second.h_;
    } else {
      return nullptr;
    }
  }

  void Evict(HillHandle** evicted_handle = nullptr) {
    // evict key in real
    if (min_level_non_empty_ < max_points_) {
      // evict item in multi level LRU
      std::list<std::string>::iterator evicted_iter;
      std::string evict_key;
      int evict_level{-1};
      // int mx_ts = 0;
      bool front = false;
      if (min_level_non_empty_ <= mru_threshold_) {
        front = true;
        int attempts = 15;
        for (int i = min_level_non_empty_; i < max_points_ && attempts; i++) {
          if (real_lru_[i].empty()) {
            continue;
          }
          // MRU
          for (auto iter = real_lru_[i].begin(); iter != real_lru_[i].end() && attempts; iter++, attempts--) {
            std::string &key = *iter;
            auto &entry = real_map_[key];
            if (entry.h_->HasRefs()) {
              continue;
            }
            evict_level = i;
            evicted_iter = iter;
            evict_key = key;
            break;
          }
          if (!attempts) {
            std::cout << "Attempt to evict failed." << '\n';
            return;
          }
          if (evict_level != -1) {
            break;
          }
          // if (real_map_[real_lru_[i].front()].insert_ts > mx_ts) {
          //   // mx_ts = real_map_[real_lru_[i].front()].insert_ts;
          //   evict_key = real_lru_[i].front();
          //   evict_level = i;
          // }
          // break;
        }
      } else {
        evict_key = real_lru_[min_level_non_empty_].back();
        evict_level = min_level_non_empty_;
      }
      if (front) {
        if (evict_level == -1) {
          std::cout << "Attempt to evict failed." << '\n';
          return;
        }
        // real_lru_[evict_level].pop_front();
        real_lru_[evict_level].erase(evicted_iter);
      } else {
        real_lru_[evict_level].pop_back();
      }
      if (evicted_handle) {
        *evicted_handle = real_map_[evict_key].h_;
      }
      real_map_.erase(evict_key);
      // move to ghost
      if (ghost_size_ != 0 && evict_level != 0) {
        MoveToGhost(evict_key, evict_level);
      }
      while (real_lru_[min_level_non_empty_].empty() &&
             min_level_non_empty_ < max_points_) {
        min_level_non_empty_++;
      }
      if (evicted_handle) {
        //// assert((*evicted_handle)->InCache());
        //// assert(evict_key == (*evicted_handle)->key().ToString());
      }
    } else {
      // evict item in top list
      int attempts = 15;
      std::string evict_key; // = top_lru_.back();
      int evict_level{-1}; // = real_map_[evict_key].insert_level;
      std::list<std::string>::iterator evicted_iter;
      assert(top_lru_.size());
      for (auto iter = top_lru_.rbegin(); iter != top_lru_.rend() && attempts; iter++, attempts--) {
        std::string &key = *iter;
        auto &entry = real_map_[key];
        if (entry.h_->HasRefs()) {
          continue;
        }
        evict_level = entry.insert_level;
        evicted_iter = iter.base();
        evict_key = key;
        break;
      }
      if (evict_level != -1) {
        top_lru_.erase(evicted_iter);
        if (evicted_handle) {
          *evicted_handle = real_map_[evict_key].h_;
        }
        real_map_.erase(evict_key);
        // move to ghost
        if (ghost_size_ != 0 && evict_level != 0) {
          MoveToGhost(evict_key, evict_level);
        }
      }
    }
  }
  void MoveToGhost(const std::string& key, int level) {
    EvictGhost();
    ghost_lru_.push_front(key);
    // NOTE: 这里应该不用写入value
    ghost_map_[key] =
        HillEntry(ghost_lru_.begin(), level, cur_ts_, false, nullptr);
  }
  void EvictGhost() {
    if (ghost_map_.size() >= ghost_size_) {
      std::string& _evict_key = ghost_lru_.back();
      ghost_map_.erase(_evict_key);
      ghost_lru_.pop_back();
    }
  }

  void Rolling() {
    for (int i = 1; i < max_points_; i++) {
      if (!real_lru_[i].empty()) {
        if (mru_threshold_ != -1) {
          real_lru_[i / 2].splice(real_lru_[i / 2].end(), real_lru_[i]);
        } else {
          real_lru_[i / 2].splice(real_lru_[i / 2].begin(), real_lru_[i]);
        }
        if (!real_lru_[i / 2].empty()) {
          min_level_non_empty_ = std::min(min_level_non_empty_, i / 2);
        }
      }
    }
    next_rolling_ts_ = cur_ts_ + cur_half_ * size_;
    prev_rolling_ts_ = cur_ts_;
    rolling_ts.push_back(cur_ts_);
    if ((int32_t)rolling_ts.size() > max_points_bits_) {
      rolling_ts.pop_front();
    }
  }

  void UpdateHalf(double cur_half) {
    cur_half_ = cur_half;
    if (cur_half_ < (double)1 / size_) {
      cur_half_ = (double)1 / size_;
    }
    if (cur_half_ > 1e14 / size_) {
      cur_half_ = 1e14 / size_;
    }
    next_rolling_ts_ = prev_rolling_ts_ + (int)(cur_half_ * size_);
  }

  void ReportAndClear(int32_t& miss_count,
                      int32_t& hit_count /*, int32_t &hit_top*/) {
    miss_count = interval_miss_count_;
    hit_count = interval_hit_count_;
    interval_miss_count_ = 0;
    interval_hit_count_ = 0;
    interval_hit_top_ = 0;
  }
  bool IsFull() { return real_map_.size() >= size_; }
  double GetCurHalf() const { return cur_half_; }
  void Remove(const std::string& key) {
    if (real_map_.count(key) != 0) {
      const std::string& evicted_key = key;
      int evicted_level = 0;
      if (!real_map_[key].h_recency) {
        std::list<std::string>::iterator hit_iter = real_map_[key].key_iter;
        int level = GetCurrentLevel(real_map_[key]);  // real_map_[key].second;
        // erase key in real, use level in real
        real_lru_[level].erase(hit_iter);
        real_map_.erase(key);
        while (real_lru_[min_level_non_empty_].empty() &&
               min_level_non_empty_ < max_points_) {
          min_level_non_empty_++;
        }
        // //// assert(min_level_non_empty_ < 10);
        evicted_level = level;
      } else {
        evicted_level = real_map_[key].insert_level;
        top_lru_.erase(real_map_[key].key_iter);
        real_map_.erase(key);
      }
      MoveToGhost(evicted_key, evicted_level);
    }
  }
  void Touch(const std::string& key) {
    if (ghost_map_.count(key) != 0) {
      auto& entry = ghost_map_[key];
      int cur_l = GetCurrentLevel(entry);
      ghost_lru_.erase(entry.key_iter);
      ghost_lru_.push_front(key);
      ghost_map_[key] =
          HillEntry(ghost_lru_.begin(), cur_l, cur_ts_, false, nullptr);
    }
  }
  int h1{}, h2{};  // debug
 private:
  int32_t GetCurrentLevel(const HillEntry& status) {
    int32_t est_level = status.insert_level;
    if (rolling_ts.empty()) {
      return est_level;
    }
    auto iter = rolling_ts.begin();
    for (size_t i = 0; i < rolling_ts.size(); i++) {
      if (status.insert_ts < *iter) {
        est_level >>= rolling_ts.size() - i;
        break;
      }
      iter++;
    }
    return est_level;
  }
  uint64_t size_;
  double init_half_;  // 初始半衰期系数
  double cur_half_;

 private:
  // 当前半衰期系数
  uint64_t lru_size_;            // LRU部分大小
  uint64_t ml_size_;             // RGC部分大小
  double hit_points_;            // 命中后得分
  int32_t max_points_bits_;      // 最高得分为 (1 << max_points_bits_) - 1
  int32_t max_points_;           // 最高得分
  int32_t min_level_non_empty_;  // 当前最小的得分
  // double ghost_size_ratio_;      // 虚缓存大小比例
  uint64_t ghost_size_;
  int32_t interval_hit_count_ = 0;       // 统计区间命中次数
  int32_t interval_miss_count_ = 0;      // 统计区间为命中次数
  int32_t interval_hit_top_ = 0;         // 在top上命中次数
  int32_t next_rolling_ts_ = INT32_MAX;  // 下一次滚动的时间戳
  int32_t cur_ts_ = 0;                   // 当前时间戳
  int32_t prev_rolling_ts_ = 0;
  std::vector<std::list<std::string>> real_lru_;
  std::list<std::string> ghost_lru_;
  std::list<std::string> top_lru_;
  std::unordered_map<std::string, HillEntry> real_map_, ghost_map_;
  std::list<int32_t> rolling_ts;
  double top_ratio_;
  int32_t mru_threshold_;
  friend class HillCache;
  friend class HillReplacer;
};

class HillReplacer : public Replacer {
 public:
  HillReplacer(int32_t buffer_size, int32_t _stats_interval = 1000,
               double init_half = 16.0f, double hit_point = 1.0f,
               int32_t max_points_bits = 6, double ghost_size_ratio = 4.0f,
               double lambda = 1.0f, double simulator_ratio = 0.67f,
               double top_ratio = 0.05f, double delta_bound = 0.01f,
               bool update_equals_size = true, int32_t mru_threshold = 64,
               int32_t minimal_update_size = 10000)
      : Replacer(buffer_size, _stats_interval),
        replacer_r_(buffer_size, init_half, hit_point, max_points_bits,
                    ghost_size_ratio, top_ratio, mru_threshold),
        replacer_s_(buffer_size, init_half * simulator_ratio, hit_point,
                    max_points_bits, ghost_size_ratio, top_ratio,
                    mru_threshold),
        lambda_(lambda),
        init_half_(init_half),
        simulator_ratio_(simulator_ratio),
        delta_bound_(delta_bound) {
    update_interval_ = std::max(100, buffer_size);
    update_interval_ = std::min(minimal_update_size, buffer_size);
  }

  RC access(const std::string& key, HillHandle* h) {
    ts++;
    if (replacer_r_.Access(key, h)) {
      increase_hit_count();
      hit_recorder.push_back(1);
      recorder_hit_count++;
    } else {
      increase_miss_count();
      hit_recorder.push_back(0);
    }

    if (hit_recorder.size() >= 500000) {
      if (hit_recorder.front() == 1) {
        recorder_hit_count--;
      }
      hit_recorder.pop_front();
    }

    replacer_s_.Access(key, nullptr);
    if (ts % update_interval_ == 0) {
      int32_t r_mc, r_hc;
      int32_t s_mc, s_hc;
      replacer_r_.ReportAndClear(r_mc, r_hc);
      replacer_s_.ReportAndClear(s_mc, s_hc);
      double r_hr = (double)r_hc / (r_mc + r_hc);
      double s_hr = (double)s_hc / (s_mc + s_hc);
      double cur_half = replacer_r_.GetCurHalf();
      if (r_hr != 0 && s_hr != 0) {
        if (fabs(s_hr - r_hr) >= EPSILON) {
          stable_count_ = 0;
          if (s_hr > r_hr) {
            double delta_ratio = (s_hr / r_hr - 1);
            if (delta_ratio > delta_bound_) {
              delta_ratio = delta_bound_;
            }
            replacer_r_.UpdateHalf(cur_half / (1 + delta_ratio * lambda_));
          } else {
            double delta_ratio = (r_hr / s_hr - 1);
            if (delta_ratio > delta_bound_) {
              delta_ratio = delta_bound_;
            }
            replacer_r_.UpdateHalf(cur_half * (1 + delta_ratio * lambda_));
          }
        } else {
          stable_count_++;
          if (stable_count_ == 5) {
            double delta_ratio = 0.1;
            if (delta_ratio > delta_bound_) {
              delta_ratio = delta_bound_;
            }
            if (cur_half < init_half_) {
              replacer_r_.UpdateHalf(cur_half * (1 + delta_ratio * lambda_));
            } else {
              replacer_r_.UpdateHalf(cur_half / (1 + delta_ratio * lambda_));
            }
            stable_count_ = 0;
          }
        }
      }
    }
    replacer_s_.UpdateHalf(replacer_r_.GetCurHalf() * simulator_ratio_);
    if (enable_interval_stats_ && ts % stats_interval == 0) {
      std::cout << stats();
    }

    return RC::SUCCESS;
  }

  RC EnsureBothReplacerFree(autovector<HillHandle*>* deleted) {
    if (replacer_r_.IsFull()) {
      HillHandle* evicted_handle;
      replacer_r_.Evict(&evicted_handle);
      //// assert(!replacer_r_.IsFull());
      //// assert(evicted_handle->InCache() && !evicted_handle->HasRefs());
      deleted->push_back(evicted_handle);
    }
    // if (replacer_s_.IsFull()) {
    //   replacer_s_.Evict();
    //   //// assert(!replacer_s_.IsFull());
    // }
    return RC::SUCCESS;
  }

  HillHandle* get(const std::string& key) {
    HillHandle* handle = replacer_r_.Get(key);
    return handle;
  }

  void Remove(const std::string& key) {
    replacer_r_.Remove(key);
    // replacer_s_.Remove(key);
    //// assert(replacer_r_.Get(key) == nullptr);
    //// assert(replacer_s_.Get(key) == nullptr);
  }
  void Touch(const std::string& key, HillHandle *e) {
    // std::cout << "touch";
    // replacer_r_.Touch(key);
    replacer_r_.Access(key, e);
    // assert(replacer_r_.ghost_map_.find(key) != replacer_r_.ghost_map_.end());
  }
  void Insert(const std::string& key, HillHandle* h) {
    //// assert(replacer_r_.Get(key) == nullptr);
    //// assert(h->InCache());
    replacer_r_.Access(key, h);
    // replacer_s_.Access(key, nullptr);
  }
  bool IsFull() { return replacer_r_.IsFull(); }
  int EvictableCount() { return replacer_r_.real_map_.size(); }
  HillHandle* EvictOne() {
    HillHandle* evicted_handle = nullptr;
    replacer_r_.Evict(&evicted_handle);
    // if (replacer_s_.real_map_.size() - replacer_s_.top_lru_.size()) {
    //   replacer_s_.Evict();
    // }
    //// assert(!replacer_r_.IsFull());
    // //// assert(evicted_handle->InCache() && !evicted_handle->HasRefs());
    return evicted_handle;
  }
  std::string get_name() { return {"Hill-Cache"}; }

 private:
  friend class HillCache;
  HillSubReplacer replacer_r_;
  HillSubReplacer replacer_s_;
  double lambda_;            // 学习率
  int32_t update_interval_;  // 更新间隔
  int32_t stable_count_{};
  double init_half_;
  double simulator_ratio_;
  double delta_bound_;
  std::list<int32_t> hit_recorder;
  int32_t recorder_hit_count{};
};

// class ALIGN_AS(CACHE_LINE_SIZE) HillCacheShard final : public CacheShardBase
// {
// };

class HillCache
#ifdef NDEBUG
    final
#endif
    : public Cache {
 public:  // functions
  virtual const char* Name() const override { return "HillCache"; }

  // 需要区分这里的buffer_size表示对象数量，默认的capacity表示占用空间数量
  // 默认块大小为 4 KB
  HillCache(uint64_t buffer_size, int32_t _stats_interval = 1000,
            double init_half = 16.0f, double hit_point = 1.0f,
            int32_t max_points_bits = 6, double ghost_size_ratio = 4.0f,
            double lambda = 1.0f, double simulator_ratio = 0.67f,
            double top_ratio = 0.05f, double delta_bound = 0.01f,
            bool update_equals_size = true, int32_t mru_threshold = 64,
            int32_t minimal_update_size = 10000,
            std::shared_ptr<MemoryAllocator> memory_allocator = nullptr,
            CacheMetadataChargePolicy metadata_charge_policy =
                kDefaultCacheMetadataChargePolicy)
      : hill_replacer_(buffer_size, _stats_interval, init_half, hit_point,
                       max_points_bits, ghost_size_ratio, lambda,
                       simulator_ratio, top_ratio, delta_bound,
                       update_equals_size, mru_threshold, minimal_update_size),
        capacity_(buffer_size),
        allocator_(memory_allocator),
        metadata_charge_policy_(metadata_charge_policy) {}

  virtual Status Insert(
      const Slice& key, ObjectPtr obj, const CacheItemHelper* helper,
      size_t charge, Handle** handle = nullptr,
      Priority priority = Priority::LOW, const Slice& compressed = Slice(),
      CompressionType type = CompressionType::kNoCompression) override {
    //
    auto&& k = key.ToString();
    // NOTE: 暂时用不到，这里把hash都设置为0
    HillHandle* e = CreateHandle(k, 0, obj, helper, charge);
    e->SetPriority(priority);
    e->SetInCache(true);
    return InsertItem(e, reinterpret_cast<HillHandle**>(handle));
  }

  Status InsertItem(HillHandle* e, HillHandle** handle) {
    Status s = Status::OK();
    autovector<HillHandle*> last_reference_list;
    // total_c++;
    {
      DMutexLock l(mutex_);
      while ((usage_ + e->total_charge) > capacity_ &&
             hill_replacer_.EvictableCount()) {
        HillHandle* old = hill_replacer_.EvictOne();
        if (old == nullptr) {
          break;
        }
        // std::cout << "OldKey: " << old->key().ToString(true) << '\n';
        //// assert(table_.find(old->key().ToString()) != table_.end());
        //// assert(old->InCache() && !old->HasRefs());
        // hill_replacer_.Remove(old->key().ToString());
        table_.erase(old->key().ToString());
        old->SetInCache(false);
        //// assert(hill_replacer_.get(old->key().ToString()) == nullptr);
        usage_ -= old->total_charge;
        last_reference_list.push_back(old);
      }

      if (usage_ + e->total_charge > capacity_ &&
          (strict_capacity_limit_ || handle == nullptr)) {
        e->SetInCache(false);
        //// assert(hill_replacer_.get(e->key().ToString()) == nullptr);
        if (handle == nullptr) {
          last_reference_list.push_back(e);
        } else {
          free(e);
          e = nullptr;
          *handle = nullptr;
          s = Status::MemoryLimit("Insert failed due to full cache");
        }
      } else {
        HillHandle* old = nullptr;
        if (table_.find(e->key().ToString()) != table_.end()) {
          old = table_[e->key().ToString()];
        }
        table_[e->key().ToString()] = e;
        usage_ += e->total_charge;

        if (old != nullptr) {
          s = Status::OkOverwritten();
          //// assert(old->InCache());
          old->SetInCache(false);
          if (!old->HasRefs()) {
            hill_replacer_.Remove(old->key().ToString());
            usage_ -= old->total_charge;
            last_reference_list.push_back(old);
          }
        }
        if (handle == nullptr) {
          hill_replacer_.Insert(e->key().ToString(), e);
          //// assert(table_.find(e->key().ToString()) != table_.end());
        } else {
          if (!e->HasRefs()) {
            e->Ref();
            hill_replacer_.Insert(e->key().ToString(), e);
          }
          *handle = e;
        }
      }
    }
    for (HillHandle* entry : last_reference_list) {
      free(entry);
    }
    return s;
  }

  virtual Handle* Lookup(const Slice& key,
                         const CacheItemHelper* helper = nullptr,
                         CreateContext* create_context = nullptr,
                         Priority priority = Priority::LOW,
                         Statistics* stats = nullptr) override {
    auto&& k = key.ToString();
    // auto e = hill_replacer_.get(k);
    HillHandle* e = nullptr;
    {
      DMutexLock l(mutex_);
      if (table_.find(key.ToString()) != table_.end()) {
        e = table_[key.ToString()];
      }
      // HillHandle* h = nullptr;
      total_c++;
      if (e != nullptr) {
        //// assert(e->InCache());
        if (!e->HasRefs()) {
          // hill_replacer_.Remove(k);
        }
        e->Ref();
        e->SetHit();
        hit_c++;
      }
    }
#ifndef NDEBUG
    if (total_c % 10000 == 0) {
      std::cout << "HillCache hit rate: " << (double)hit_c / total_c
                << " R: " << hill_replacer_.replacer_r_.GetCurHalf() << '\n';
    }
#endif
    return reinterpret_cast<Handle*>(e);
  }

  using Cache::Release;
  bool Release(Handle* handle, bool erase_if_last_ref) override {
    auto e = reinterpret_cast<HillHandle*>(handle);
    if (e == nullptr) {
      return false;
    }
    bool must_free;
    bool was_in_cache;
    {
      DMutexLock l(mutex_);
      must_free = e->Unref();
      was_in_cache = e->InCache();
      if (must_free && was_in_cache) {
        if (usage_ > capacity_ || erase_if_last_ref) {
          table_.erase(e->key().ToString());
          e->SetInCache(false);
          //// assert(hill_replacer_.get(e->key().ToString()) == nullptr);
        } else {
          // hill_replacer_.Touch(e->key().ToString(), e);
          //// assert(e->InCache());
          //// assert(table_.find(e->key().ToString()) != table_.end());
          must_free = false;
        }
      }
      if (must_free) {
        // free(e);
        // //// assert(usage_ >= e->total_charge);
        usage_ -= e->total_charge;
        hill_replacer_.Remove(e->key().ToString());
        // std::cout << usage_.load() << '\n';
      } else {
        hill_replacer_.Touch(e->key().ToString(), e);
      }
    }
    // Free the entry here outside of mutex for performance reasons.
    if (must_free) {
      free(e);
    }
    return must_free;
  }

  virtual bool Ref(Handle* handle) override {
    DMutexLock l(mutex_);
    auto h = reinterpret_cast<HillHandle*>(handle);
    //// assert(h->HasRefs());
    h->Ref();
    return true;
  }

  virtual ObjectPtr Value(Handle* handle) override {
    auto h = reinterpret_cast<const HillHandle*>(handle);
    return h->value;
  }

  virtual size_t GetCharge(Handle* handle) const override {
    return reinterpret_cast<const HillHandle*>(handle)->GetCharge(
        metadata_charge_policy_);
  }

  virtual void Erase(const Slice& key) override {
    std::cout << "Unsupported!";
    abort();
  };

  virtual Handle* CreateStandalone(const Slice& key, ObjectPtr obj,
                                   const CacheItemHelper* helper, size_t charge,
                                   bool allow_uncharged) override {
    std::cout << "Unsupported!";
    abort();
    return nullptr;
  }

  virtual uint64_t NewId() override { return 0; }

  virtual void SetCapacity(size_t capacity) override {
    std::cout << "Unsupported!";
    abort();
  }

  virtual void SetStrictCapacityLimit(bool strict_capacity_limit) override {
    std::cout << "Unsupported!";
    abort();
  }

  virtual bool HasStrictCapacityLimit() const override {
    std::cout << "Unsupported!";
    abort();
    return false;
  }

  virtual size_t GetCapacity() const override { return capacity_; }

  virtual size_t GetUsage() const override { return usage_; }

  virtual size_t GetUsage(Handle* handle) const override {
    return reinterpret_cast<const HillHandle*>(handle)->GetCharge(
        metadata_charge_policy_);
  }

  virtual size_t GetPinnedUsage() const override {
    std::cout << "Unsupported!";
    abort();
    return 0;
  }

  virtual const CacheItemHelper* GetCacheItemHelper(
      Handle* handle) const override {
    return reinterpret_cast<const HillHandle*>(handle)->helper;
  }

  // FIXME:
  virtual void ApplyToAllEntries(
      const std::function<void(const Slice& key, ObjectPtr obj, size_t charge,
                               const CacheItemHelper* helper)>& callback,
      const ApplyToAllEntriesOptions& opts) override {
    std::cout << "ApplyToAllEntries Unsupported!\n";
    // abort();
  }

  virtual void EraseUnRefEntries() override {
    std::cout << "Unsupporteded!";
    abort();
  }
  double GetHitRate() { return (double)hit_c / total_c; }
  bool strict_capacity_limit_ = false;

 private:
  HillHandle* CreateHandle(const std::string& key, uint32_t hash,
                           Cache::ObjectPtr value,
                           const Cache::CacheItemHelper* helper,
                           size_t charge) {
    //// assert(helper);
    // value == nullptr is reserved for indicating failure in SecondaryCache
    //// assert(!(helper->IsSecondaryCacheCompatible() && value == nullptr));

    // Allocate the memory here outside of the mutex.
    // If the cache is full, we'll have to release it.
    // It shouldn't happen very often though.
    HillHandle* e =
        static_cast<HillHandle*>(malloc(sizeof(HillHandle) - 1 + key.size()));

    e->key_length = key.size();
    e->value = value;
    e->m_flags = 0;
    e->im_flags = 0;
    e->helper = helper;
    e->key_length = key.size();
    e->hash = hash;
    e->refs = 0;
    memcpy(e->key_data, key.data(), key.size());
    e->CalcTotalCharge(charge, metadata_charge_policy_);

    return e;
  }
  HillReplacer hill_replacer_;
  uint64_t capacity_;
  mutable DMutex mutex_;
  std::shared_ptr<MemoryAllocator> const allocator_;
  std::atomic<uint64_t> usage_;
  CacheMetadataChargePolicy metadata_charge_policy_;
  uint64_t hit_c = 0;
  uint64_t total_c = 0;
  std::unordered_map<std::string, HillHandle*> table_;
};

}  // namespace hill

}  // namespace ROCKSDB_NAMESPACE
