
> Diff after splitting based on type
```
diff <(cat data/asr_data/call_upwork_test_cnd_*/manifest.json |sort) <(cat data/asr_data/call_upwork_test_cnd/manifest.json |sort)
```

> Prepare Augmented Data
```
plume data filter /dataset/png_entities/png_numbers_2020_07/ /dataset/png_entities/png_numbers_2020_07_skip1hour/

plume data augment /dataset/agara_slu/call_alphanum_ag_sg_v1_abs/ /dataset/png_entities/png_numbers_2020_07_1hour_noblank/ /dataset/png_entities/png_numbers_2020_07_skip1hour/ /dataset/png_entities/aug_pngskip1hour-agsgalnum-1hournoblank/

plume data filter --kind transform_digits /dataset/agara_slu/png1hour-agsgalnum-1hournoblank/ /dataset/agara_slu/png1hour-agsgalnum-1hournoblank_prep/
```


```
KENLM_INC=/usr/local/include/kenlm/ pip install -e ../deps/wav2letter/bindings/python/
```
