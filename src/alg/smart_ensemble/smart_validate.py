
# Two kind of merging
# Just take the best method
# Or
# Weight al the methods and use borda count to build final
import os

import pickle

from src.alg.smart_ensemble.intialize_sets import build_preds
from src.data import Cache
from src.metrics import ap_at_k

path = os.path.dirname(os.path.realpath(__file__)) + "/validate"


cache = Cache()
targets = cache.fetch("targets")


def validate_and_build_preds(model_names, k=5, mode='max'):

    models_preds_over_k = {}

    for i in range(k):

        rel_path = path + "/" + str(i)
        rel_path_preds = rel_path + "/preds"
        rel_path_sets = rel_path + "/sets"

        models_preds = {}

        for model_name in model_names:
            with open(rel_path_sets + "/test.obj", "rb") as f:
                test_set = pickle.load(f)

            preds = build_preds(rel_path_preds, model_name)
            models_preds[model_name] = [preds, test_set]

        models_preds_over_k[i] = models_preds

    if mode == 'max':
        with open(os.path.dirname(os.path.realpath(__file__)) + '/smart_max.csv', 'w') as f:

            final_results = {}

            for playlist in targets:
                print("Processing playlist number: " + str(playlist))

                best_and_model = (-1, None)

                for model in model_names:

                    tot_ap = 0
                    for i in range(k):
                        model_preds, test_set = models_preds_over_k[i][model]
                        model_ap_at_k = ap_at_k(model_preds[playlist], test_set[playlist])
                        tot_ap += model_ap_at_k

                    tot_ap /= k

                    best_and_model = (tot_ap, model) if tot_ap > best_and_model[0] else best_and_model

                final_results[playlist] = best_and_model[1]

            for playlist, model in final_results.items():
                f.write(str(playlist) + ",")
                f.write(str(model))
                f.write("\n")

    else:

        with open(os.path.dirname(os.path.realpath(__file__)) + '/smart_borda.csv', 'w') as f:

            final_results = {}

            for playlist in targets:
                print("Processing playlist number: " + str(playlist))

                model_aps = []

                for model in model_names:

                    tot_ap = 0
                    for i in range(k):
                        model_preds, test_set = models_preds_over_k[i][model]
                        model_ap_at_k = ap_at_k(model_preds[playlist], test_set[playlist])
                        tot_ap += model_ap_at_k

                    tot_ap /= k
                    model_aps.append((model, tot_ap))

                tot = sum(ap for _, ap in model_aps)

                if tot != 0:
                    model_w = [(model, ap/tot) for model, ap in model_aps]
                else:
                    model_w = [(model, 1) for model, ap in model_aps]

                final_results[playlist] = model_w

            for playlist, model in final_results.items():
                f.write(str(playlist) + ",")
                model_w = final_results[playlist]

                for m, w in model_w:
                    f.write(str(m) + "=" + str(w))
                    f.write("-")

                f.write("\n")






