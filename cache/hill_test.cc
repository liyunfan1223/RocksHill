#include "cache/hill.h"

#include <cstdint>
#include <forward_list>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "cache/lru_cache.h"
#include "cache/typed_cache.h"
#include "port/stack_trace.h"
#include "rocksdb/cache.h"
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"
#include "rocksdb/table.h"
#include "rocksdb/utilities/options_util.h"
#include "test_util/secondary_cache_test_util.h"
#include "test_util/testharness.h"
#include "util/coding.h"
#include "util/hash_containers.h"
#include "util/string_util.h"

namespace ROCKSDB_NAMESPACE {

namespace {

// Conversions between numeric keys/values and the types expected by Cache.
std::string EncodeKey16Bytes(int k) {
  std::string result;
  PutFixed32(&result, k);
  result.append(std::string(12, 'a'));  // Because we need a 16B output, we
                                        // add a 12-byte padding.
  return result;
}

int DecodeKey16Bytes(const Slice& k) {
  assert(k.size() == 16);
  return DecodeFixed32(k.data());  // Decodes only the first 4 bytes of k.
}

std::string EncodeKey32Bits(int k) {
  std::string result;
  PutFixed32(&result, k);
  return result;
}

int DecodeKey32Bits(const Slice& k) {
  assert(k.size() == 4);
  return DecodeFixed32(k.data());
}

Cache::ObjectPtr EncodeValue(uintptr_t v) {
  return reinterpret_cast<Cache::ObjectPtr>(v);
}

int DecodeValue(void* v) {
  return static_cast<int>(reinterpret_cast<uintptr_t>(v));
}

const Cache::CacheItemHelper kDumbHelper{
    CacheEntryRole::kMisc,
    [](Cache::ObjectPtr /*value*/, MemoryAllocator* /*alloc*/) {}};

const Cache::CacheItemHelper kInvokeOnDeleteHelper{
    CacheEntryRole::kMisc,
    [](Cache::ObjectPtr value, MemoryAllocator* /*alloc*/) {
      auto& fn = *static_cast<std::function<void()>*>(value);
      fn();
    }};
}  // anonymous namespace

class CacheTest : public testing::Test,
                  public secondary_cache_test_util::WithCacheTypeParam {
 public:
  static CacheTest* current_;
  static std::string type_;

  static void Deleter(Cache::ObjectPtr v, MemoryAllocator*) {
    current_->deleted_values_.push_back(DecodeValue(v));
  }
  static const Cache::CacheItemHelper kHelper;

  static const int kCacheSize = 1000;
  static const int kNumShardBits = 4;

  static const int kCacheSize2 = 100;
  static const int kNumShardBits2 = 2;

  std::vector<int> deleted_values_;
  std::shared_ptr<Cache> cache_;

  CacheTest() : cache_(NewCache(kCacheSize, kNumShardBits, false)) {
    current_ = this;
    type_ = GetParam();
  }

  ~CacheTest() override {}

  // These functions encode/decode keys in tests cases that use
  // int keys.
  // Currently, HyperClockCache requires keys to be 16B long, whereas
  // LRUCache doesn't, so the encoding depends on the cache type.
  std::string EncodeKey(int k) {
    if (IsHyperClock()) {
      return EncodeKey16Bytes(k);
    } else {
      return EncodeKey32Bits(k);
    }
  }

  int DecodeKey(const Slice& k) {
    if (IsHyperClock()) {
      return DecodeKey16Bytes(k);
    } else {
      return DecodeKey32Bits(k);
    }
  }

  int Lookup(std::shared_ptr<Cache> cache, int key) {
    Cache::Handle* handle = cache->Lookup(EncodeKey(key));
    const int r = (handle == nullptr) ? -1 : DecodeValue(cache->Value(handle));
    if (handle != nullptr) {
      cache->Release(handle);
    }
    return r;
  }

  void Insert(std::shared_ptr<Cache> cache, int key, int value,
              int charge = 1) {
    EXPECT_OK(cache->Insert(EncodeKey(key), EncodeValue(value), &kHelper,
                            charge, /*handle*/ nullptr, Cache::Priority::HIGH));
  }

  void Erase(std::shared_ptr<Cache> cache, int key) {
    cache->Erase(EncodeKey(key));
  }

  int Lookup(int key) { return Lookup(cache_, key); }

  void Insert(int key, int value, int charge = 1) {
    Insert(cache_, key, value, charge);
  }

  void Erase(int key) { Erase(cache_, key); }

  static constexpr uint64_t kKilobyte = 1024;
  static constexpr uint64_t kMegabyte = kKilobyte * kKilobyte;
  static constexpr uint64_t kGigabyte = kKilobyte * kMegabyte;

  static constexpr uint64_t kDefaultBlockSize = 4 * kKilobyte;
};

const Cache::CacheItemHelper CacheTest::kHelper{CacheEntryRole::kMisc,
                                                &CacheTest::Deleter};

CacheTest* CacheTest::current_;
std::string CacheTest::type_;

#if 0
TEST_P(CacheTest, UsageTest) {
  // cache is std::shared_ptr and will be automatically cleaned up.
  const size_t kCapacity = 100000;
  auto cache = NewCache(kCapacity, 6, false, kDontChargeCacheMetadata);
  auto precise_cache = NewCache(kCapacity, 0, false, kFullChargeCacheMetadata);
  ASSERT_EQ(0, cache->GetUsage());
  size_t baseline_meta_usage = precise_cache->GetUsage();
  if (!IsHyperClock()) {
    ASSERT_EQ(0, baseline_meta_usage);
  }

  size_t usage = 0;
  char value[10] = "abcdef";
  // make sure everything will be cached
  for (int i = 1; i < 100; ++i) {
    std::string key = EncodeKey(i);
    auto kv_size = key.size() + 5;
    ASSERT_OK(cache->Insert(key, value, &kDumbHelper, kv_size));
    ASSERT_OK(precise_cache->Insert(key, value, &kDumbHelper, kv_size));
    usage += kv_size;
    ASSERT_EQ(usage, cache->GetUsage());
    if (GetParam() == kFixedHyperClock) {
      ASSERT_EQ(baseline_meta_usage + usage, precise_cache->GetUsage());
    } else {
      // AutoHyperClockCache meta usage grows in proportion to lifetime
      // max number of entries. LRUCache in proportion to resident number of
      // entries, though there is an untracked component proportional to
      // lifetime max number of entries.
      ASSERT_LT(usage, precise_cache->GetUsage());
    }
  }

  // cache->EraseUnRefEntries();
  // precise_cache->EraseUnRefEntries();
  // ASSERT_EQ(0, cache->GetUsage());
  // if (GetParam() != kAutoHyperClock) {
  //   // NOTE: AutoHyperClockCache meta usage grows in proportion to lifetime
  //   // max number of entries.
  //   ASSERT_EQ(baseline_meta_usage, precise_cache->GetUsage());
  // }

  // make sure the cache will be overloaded
  // for (size_t i = 1; i < kCapacity; ++i) {
  //   std::string key = EncodeKey(static_cast<int>(1000 + i));
  //   ASSERT_OK(cache->Insert(key, value, &kDumbHelper, key.size() + 5));
  //   ASSERT_OK(precise_cache->Insert(key, value, &kDumbHelper, key.size() + 5));
  // }

  // the usage should be close to the capacity
  // ASSERT_GT(kCapacity, cache->GetUsage());
  // ASSERT_GT(kCapacity, precise_cache->GetUsage());
  // ASSERT_LT(kCapacity * 0.95, cache->GetUsage());
  // if (!IsHyperClock()) {
  //   ASSERT_LT(kCapacity * 0.95, precise_cache->GetUsage());
  // } else {
  //   // estimated value size of 1 is weird for clock cache, because
  //   // almost all of the capacity will be used for metadata, and due to only
  //   // using power of 2 table sizes, we might hit strict occupancy limit
  //   // before hitting capacity limit.
  //   ASSERT_LT(kCapacity * 0.80, precise_cache->GetUsage());
  // }
}


TEST_P(CacheTest, HitAndMiss) {
  ASSERT_EQ(-1, Lookup(100));

  Insert(100, 101);
  ASSERT_EQ(101, Lookup(100));
  ASSERT_EQ(-1, Lookup(200));
  ASSERT_EQ(-1, Lookup(300));

  Insert(200, 201);
  ASSERT_EQ(101, Lookup(100));
  ASSERT_EQ(201, Lookup(200));
  ASSERT_EQ(-1, Lookup(300));

  Insert(100, 102);
  if (IsHyperClock()) {
    // ClockCache usually doesn't overwrite on Insert
    ASSERT_EQ(101, Lookup(100));
  } else {
    ASSERT_EQ(102, Lookup(100));
  }
  ASSERT_EQ(201, Lookup(200));
  ASSERT_EQ(-1, Lookup(300));

  // ASSERT_EQ(1U, deleted_values_.size());
  // if (IsHyperClock()) {
  //   ASSERT_EQ(102, deleted_values_[0]);
  // } else {
  //   ASSERT_EQ(101, deleted_values_[0]);
  // }
}
#endif

// NOTE: 测试在RocksDB里的HillCache
TEST_P(CacheTest, InRocksDB) {
  DB* db;
  Options options;
  // Optimize RocksDB. This is the easiest way to get RocksDB to perform well
  options.IncreaseParallelism();
  options.OptimizeLevelStyleCompaction();
  // create the DB if it's not already present
  options.create_if_missing = true;
  options.use_direct_reads = true;
  options.use_direct_io_for_flush_and_compaction = true;

  // NOTE: 创建HillCache
  HillCacheOptions hill_opt;
  // hill_opt.capacity = 500ul << 1;
  hill_opt.capacity = 500ul << 4;
  std::shared_ptr<Cache> cache = hill_opt.MakeHillCache();
  BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));

  // open DB
  std::string kDBPath = "rocksdb_simple_example";
  rocksdb::DestroyDB(kDBPath, rocksdb::Options());
  Status s = DB::Open(options, kDBPath, &db);
  assert(s.ok());

  size_t key_num = 10000;
  for (size_t i = 0; i < key_num; i++) {
    s = db->Put(WriteOptions(), std::to_string(i), std::to_string(i));
    ASSERT_OK(s);
  }
  db->Flush(FlushOptions());

  std::string value;
  for (size_t i = 0; i < key_num; i++) {
    s = db->Get(ReadOptions(), std::to_string(i), &value);
    ASSERT_OK(s);
    ASSERT_EQ(value, std::to_string(i));
  }

  delete db;
}

TEST_P(CacheTest, InRocksDBBasicHill) {
  DB* db;
  Options options;
  // Optimize RocksDB. This is the easiest way to get RocksDB to perform well
  options.IncreaseParallelism();
  options.OptimizeLevelStyleCompaction();
  // create the DB if it's not already present
  options.create_if_missing = true;
  options.use_direct_reads = true;
  options.use_direct_io_for_flush_and_compaction = true;

  // NOTE: 创建HillCache
  HillCacheOptions hill_opt;
  // hill_opt.capacity = 500ul << 1;
  hill_opt.capacity = 16 * kMegabyte / kDefaultBlockSize;
  std::shared_ptr<Cache> cache = hill_opt.MakeHillCache();
  BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));

  // open DB
  std::string kDBPath = "rocksdb_simple_example";
  rocksdb::DestroyDB(kDBPath, rocksdb::Options());
  Status s = DB::Open(options, kDBPath, &db);
  assert(s.ok());

  size_t key_num = 10000;
  for (size_t i = 0; i < key_num; i++) {
    s = db->Put(WriteOptions(), std::to_string(i), std::to_string(i));
    ASSERT_OK(s);
  }
  db->Flush(FlushOptions());

  std::string value;
  for (size_t i = 0; i < key_num; i++) {
    s = db->Get(ReadOptions(), std::to_string(i), &value);
    ASSERT_OK(s);
    ASSERT_EQ(value, std::to_string(i));
  }

  delete db;
}

// NOTE: 测试在RocksDB里的HillCache
TEST_P(CacheTest, InRocksDBBasicLRU) {
  DB* db;
  Options options;
  // Optimize RocksDB. This is the easiest way to get RocksDB to perform well
  options.IncreaseParallelism();
  options.OptimizeLevelStyleCompaction();
  // create the DB if it's not already present
  options.create_if_missing = true;
  options.use_direct_reads = true;
  options.use_direct_io_for_flush_and_compaction = true;

  LRUCacheOptions lru_opt;
  // hill_opt.capacity = 500ul << 1;
  lru_opt.capacity = 16 * kMegabyte;
  lru_opt.num_shard_bits = 0;
  std::shared_ptr<Cache> cache = lru_opt.MakeSharedCache();
  BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));

  // open DB
  std::string kDBPath = "rocksdb_simple_example";
  rocksdb::DestroyDB(kDBPath, rocksdb::Options());
  Status s = DB::Open(options, kDBPath, &db);
  assert(s.ok());

  size_t key_num = 10000;
  for (size_t i = 0; i < key_num; i++) {
    s = db->Put(WriteOptions(), std::to_string(i), std::to_string(i));
    ASSERT_OK(s);
  }
  db->Flush(FlushOptions());

  std::string value;
  for (size_t i = 0; i < key_num; i++) {
    s = db->Get(ReadOptions(), std::to_string(i), &value);
    ASSERT_OK(s);
    ASSERT_EQ(value, std::to_string(i));
  }

  delete db;
}

INSTANTIATE_TEST_CASE_P(CacheTestInstance, CacheTest,
                        testing::Values(secondary_cache_test_util::kHillCache));

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  std::cout << "Test";
  return RUN_ALL_TESTS();
}