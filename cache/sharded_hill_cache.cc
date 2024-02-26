// //
// // Created by Yunfan Li on 2023/5/23.
// //

#include "cache/sharded_hill_cache.h"

#include "cache/hill.h"
#include "rocksdb/cache.h"

namespace ROCKSDB_NAMESPACE {

std::shared_ptr<Cache> ShardedHillCacheOptions::MakeHillCache() const {
  auto opt = *this;
  if (opt.num_shard_bits < 0) {
    opt.num_shard_bits = 2;
  }
  auto hill = std::make_shared<sharded_hill_cache::ShardedHillCache>(opt);
  return hill;
}

}  // namespace ROCKSDB_NAMESPACE
