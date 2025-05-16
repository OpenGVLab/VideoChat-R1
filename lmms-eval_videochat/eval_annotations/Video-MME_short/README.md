---
dataset_info:
  config_name: videomme_short
  features:
  - name: video_id
    dtype: string
  - name: duration
    dtype: string
  - name: domain
    dtype: string
  - name: sub_category
    dtype: string
  - name: url
    dtype: string
  - name: videoID
    dtype: string
  - name: question_id
    dtype: string
  - name: task_type
    dtype: string
  - name: question
    dtype: string
  - name: options
    sequence: string
  - name: answer
    dtype: string
  splits:
  - name: test
    num_bytes: 1003241.0
    num_examples: 900
  download_size: 405167
  dataset_size: 1003241.0
configs:
- config_name: videomme_short
  data_files:
  - split: test
    path: videomme_short/test-*
---
