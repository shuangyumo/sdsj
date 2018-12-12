import argparse
from libs.automl import AutoML

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--train-csv',type=argparse.FileType('r'), required=True)
    parser.add_argument('--test-csv',type=argparse.FileType('r'), required=True)
    parser.add_argument('--prediction-csv', type=argparse.FileType('w'), required=True)
    args = parser.parse_args()

    automl = AutoML(args.model_dir)

    automl.train(args.train_csv, args.mode)
    automl.save()
    automl.load()
    automl.predict(args.test_csv, args.prediction_csv)


if __name__ == '__main__':
    main()