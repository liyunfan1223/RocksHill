// //
// // Created by Yunfan Li on 2023/5/23.
// //

#include "cache/hill.h"

#include "rocksdb/cache.h"

namespace ROCKSDB_NAMESPACE {

std::shared_ptr<Cache> HillCacheOptions::MakeHillCache() const {
  auto hill = std::make_shared<hill::HillCache>(
      capacity, stats_interval, init_half, hit_point, max_points_bits,
      ghost_size_ratio, lambda, simulator_ratio, top_ratio, delta_bound,
      update_equals_size, mru_threshold, minimal_update_size, memory_allocator,
      metadata_charge_policy);
  return hill;
}

}  // namespace ROCKSDB_NAMESPACE
