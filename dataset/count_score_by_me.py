import json
import os

def count_score(gt_pth="./val_gt.json", pred_pth="./val_predictions.json"):

    with open(gt_pth, 'r') as f:
        gt_data = json.load(f)

    with open(pred_pth, 'r') as f:
        pred_data = json.load(f)

    top1_correct = 0
    top3_correct = 0

    for k in list(pred_data.keys()):
        print(k)
        ans = gt_data[k]
        pred = pred_data[k]
        if ans == pred[0]:
            top1_correct += 1
            top3_correct += 1

        elif ans in pred[:3]:
            top3_correct += 1

    total = len(pred_data)
    top1_acc = top1_correct / total
    top3_acc = top3_correct / total

    print(f"Top-1 Accuracy: {top1_acc*100:.2f}% ({top1_correct}/{total})")
    print(f"Top-3 Accuracy: {top3_acc*100:.2f}% ({top3_correct}/{total})")

if __name__ == "__main__":
    count_score()