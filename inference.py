import evaluate
import deployment.utils as dutils


def local_predict():

    args = evaluate.get_arg_parser()

    lb, sft = dutils.predict(args.data_dir, args)

    print(lb)
    print(sft)

if __name__ == '__main__':
    local_predict()