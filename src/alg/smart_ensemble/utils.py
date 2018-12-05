import os
from src.data import Cache

cache = Cache()


def write_preds_to_file(name, model, dataset, targets=cache.fetch("targets")):

    if dataset == "train_set":
        path = os.path.dirname(os.path.realpath(__file__)) + "/preds_test"
    else:
        path = os.path.dirname(os.path.realpath(__file__)) + "/preds"

    preds = model.run(dataset=dataset, targets=targets)

    # Create directory if necessary
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, name + ".csv"), 'w') as f:
        f.write("playlist_id,track_ids\n")
        for row in preds:
            f.write(str(row[0]) + ",")
            f.write(" ".join([str(el) for el in row[1]]))
            f.write("\n")


def build_preds_from_file(filename, test=False):

    if test:
        path = os.path.dirname(os.path.realpath(__file__)) + "/preds_test/"
    else:
        path = os.path.dirname(os.path.realpath(__file__)) + "/preds/"

    preds = {}

    with open(os.path.join(path, filename + ".csv"), 'r') as f:

        lines = f.readlines()[1:]

        for line in lines:
            playlist, tracks = line.split(",")
            playlist = int(playlist)
            tracks = [int(x) for x in tracks.split(" ")]
            preds[playlist] = tracks

    return preds



