
import argparse, json
import pandas as pd
from .utils import load_obj
from .features import ALL_FEATURES

EXAMPLE = {
    "sex": 2, "education": 2, "marriage": 1,
    "pay_0": 0, "pay_2": 0, "pay_3": 0, "pay_4": 0, "pay_5": 0, "pay_6": 0,
    "limit_bal": 200000, "age": 35,
    "bill_amt1": 3913, "bill_amt2": 3102, "bill_amt3": 689, "bill_amt4": 0, "bill_amt5": 0, "bill_amt6": 0,
    "pay_amt1": 0, "pay_amt2": 689, "pay_amt3": 0, "pay_amt4": 0, "pay_amt5": 0, "pay_amt6": 0
}

def main(args):
    model = load_obj(args.model_path)

    if args.json is None:
        payload = EXAMPLE
    else:
        payload = json.loads(args.json)

    df = pd.DataFrame([payload], columns=ALL_FEATURES)
    proba = float(model.predict_proba(df)[:,1][0])
    pred = int(proba >= args.threshold)
    print(json.dumps({"probability_default": proba, "prediction": pred, "threshold": args.threshold}, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/model.pkl")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--json", type=str, default=None, help='JSON string with feature values')
    args = parser.parse_args()
    main(args)
