# Changelog

## [0.1.1](https://github.com/justinrmiller/lancedb-ray/compare/v0.1.0...v0.1.1) (2026-09-23)


### Bug Fixes

* **io:** size the upsert key shuffle by the transaction budget ([#6](https://github.com/justinrmiller/lancedb-ray/issues/6)) ([72d0a6c](https://github.com/justinrmiller/lancedb-ray/commit/72d0a6cd7f98448af302aeb6b94f75c959fd6c25))

## 0.1.0 (2026-09-23)


### Features

* **benchmarks:** add a benchmark suite that gates on correctness, not timings ([8cfb7b6](https://github.com/justinrmiller/lancedb-ray/commit/8cfb7b64423ad43d1696e975749b9753e9df5a64))
* bound a write task by bytes, and expose the scan and file knobs ([1f59d7f](https://github.com/justinrmiller/lancedb-ray/commit/1f59d7fcd656af2b554d01cf5ebd21e855564502))
* **examples:** add a vLLM generate-embed-search pipeline ([9c613ab](https://github.com/justinrmiller/lancedb-ray/commit/9c613ab1e620de3fba2d19e6a0dda8968007a228))
* **examples:** store image bytes in the table and add a row inspector ([87925c5](https://github.com/justinrmiller/lancedb-ray/commit/87925c5e7c6daa8196970b38b85e269ca4e1f3de))
* Ray Data integration for LanceDB and LanceDB Enterprise ([07ddca9](https://github.com/justinrmiller/lancedb-ray/commit/07ddca9a5ee412e2487b1f3a98e903fa86f11fb0))


### Bug Fixes

* an empty overwrite must empty the table, not keep the old rows ([3ff1eca](https://github.com/justinrmiller/lancedb-ray/commit/3ff1ecadd2ea0eca9a2eb0241edc7b49f248c9f6))
* bound the per-process connection cache ([8cc838c](https://github.com/justinrmiller/lancedb-ray/commit/8cc838c5bcf54e623508e278b941511ece145425))
* create an empty table from the dataset's own schema on the fragment path ([ffd3607](https://github.com/justinrmiller/lancedb-ray/commit/ffd3607721983d70c552919029cb0c910b154fa1))
* **examples:** create the bucket and clear the prefix before each S3 run ([beb13b2](https://github.com/justinrmiller/lancedb-ray/commit/beb13b234533724f26276cda2655db7eea75e182))
* **examples:** load CLIP by class name to avoid a torchvision import ([be57452](https://github.com/justinrmiller/lancedb-ray/commit/be57452d748a6424c1446b08b3c7d42f9e00bbba))
* **examples:** pin Ray to 2.58.0 and carry the prompt past chat templating ([18f7804](https://github.com/justinrmiller/lancedb-ray/commit/18f7804bca47f73ed4ec47c77b4900e4d23e0263))
* fail an upsert early when Ray cannot hash-partition on the key ([436ab04](https://github.com/justinrmiller/lancedb-ray/commit/436ab045809d7dfc0e6e19c0875db44f18e6ccd4))
* honour per_task_row_limit in the single-task read strategy ([56a3b9e](https://github.com/justinrmiller/lancedb-ray/commit/56a3b9eaea245fa2efd4af86f6771a9b4c7f9970))
* keep fragment options on the empty-input path, and correct the memory claim ([54602f4](https://github.com/justinrmiller/lancedb-ray/commit/54602f43a3f8d77b808787ca8ac8fab765d4bdc5))
* make batch_size mean something on a local read ([89451c9](https://github.com/justinrmiller/lancedb-ray/commit/89451c963439f338bc7a247f89efca54c3865db6))
* match HTTP status codes as whole numbers, not as digits anywhere ([945bfa8](https://github.com/justinrmiller/lancedb-ray/commit/945bfa823e87b95039a0db1eaa46f859f1575abe))
* name concurrency in its own error instead of Ray's internals ([5898cb1](https://github.com/justinrmiller/lancedb-ray/commit/5898cb122b14e395b8de275ccee6b30850020fe9))
* never overwrite a completed write when the post-write check fails ([b287b74](https://github.com/justinrmiller/lancedb-ray/commit/b287b746352bb7c63c4c200eee84b95e9f4d52e9))
* never stand an empty table where rows went missing ([4326b54](https://github.com/justinrmiller/lancedb-ray/commit/4326b54b52361552270711171093daa00996fa25))
* pin the table version before sizing the remote scan ([7390d4c](https://github.com/justinrmiller/lancedb-ray/commit/7390d4c26352776317037397751a1fbe447fe2d6))
* reject columns=[] instead of silently reading every column ([b1513ea](https://github.com/justinrmiller/lancedb-ray/commit/b1513eabd1df2e12a9ef844f6c34d30181e7b25e))
* rotated credentials and unaligned Arrow buffers from Ray blocks ([58fb200](https://github.com/justinrmiller/lancedb-ray/commit/58fb200518b6b28bdfb8711f58826393a98cb9ad))
* stop retrying appends that may already have committed ([406ab50](https://github.com/justinrmiller/lancedb-ray/commit/406ab505849ae0c81ddcca6d637cc8b65f95ebe9))
* two correctness bugs found by an adversarial pass ([9596f7d](https://github.com/justinrmiller/lancedb-ray/commit/9596f7dc229f6202de0c05856665aa4785c444e0))
* validate table names and create a table from an empty write ([948c384](https://github.com/justinrmiller/lancedb-ray/commit/948c3845d52131556f6ccb25396077ec0d2ee505))


### Performance

* cut redundant round trips on the read and write paths ([5b68d0d](https://github.com/justinrmiller/lancedb-ray/commit/5b68d0d65f4260ffbb38346c40320863fdce29d8))
* one transaction per write task, and guard parallel upserts ([e91b28e](https://github.com/justinrmiller/lancedb-ray/commit/e91b28e8b6e26b9a53afb03edce22cb92be8b401))
* stream remote read batches instead of buffering whole shards ([d08f8dc](https://github.com/justinrmiller/lancedb-ray/commit/d08f8dca222252df0b24133dbd641ab2c1918dcf))


### Documentation

* correct install instructions ([e048694](https://github.com/justinrmiller/lancedb-ray/commit/e0486945de45f5d91057037009a6e8f1be34618c))
* correct write docstrings that no longer matched the code ([4724ade](https://github.com/justinrmiller/lancedb-ray/commit/4724ade04435a64579c18805ac6ecafd9656dd01))
* give each example its own directory, add CLIP image search ([c45bab6](https://github.com/justinrmiller/lancedb-ray/commit/c45bab60b56f6a62379f739d2d01b650372fe028))
