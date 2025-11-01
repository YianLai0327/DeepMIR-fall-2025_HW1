# Data structure

```
artist20/
├── train.json
├── val.json
├── train_val/
│   ├── singer_1/
│   │   ├── album_1/
│   │   │   ├── song_1.mp3
│   │   │   ├── song_2.mp3
│   │   │   └── ...
│   │   ├── ...
│   │   └── album_5/
│   ├── ...
│   └── singer_20/
│
└── test/
    ├── 001.mp3
    ├── ...
    └── 233.mp3
```

# Some things you need to know

- Use the file in `train.json` for training, and so as `val.json` for validation.

- DON'T USE audio in `test` folder for training.

You need to follow format as `test_pred.json` for your output. 

The format of answer is like `test_ans.json`
```
[
    label_1,
    label_2,
    ...
    label_id
]
```

The format of your output should be like `test_pred.json`
```
{
    '1': [top1_pred, top2_pred, top3_pred],
    '2': [top1_pred, top2_pred, top3_pred],
    ...
    'id': [top1_pred, top2_pred, top3_pred]
}
```

I'll use `count_score.py` to calculate your final results.

```
python count_score.py [answer_path] [your_output_path]

# Example
python count_score.py ./test_ans.json ./test_pred.json
```