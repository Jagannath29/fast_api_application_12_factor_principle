# bank_note_prediction_using_fast_api documentation!

## Description

This is a simple fast_api demo

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://s3://my-bucket/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://s3://my-bucket/data/` to `data/`.


