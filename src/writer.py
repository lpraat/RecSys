"""
This file contains function used to write to disk
and create submission files
"""

import datetime
import os


def create_submission(name, recs):
    """
    Builds the submission file from the recommendations matrix

    Parameters
    ---------------
    name : string
        Prefix of the submission file
    recs : list
        List of computed recommendations
    """

    submissions_path = os.path.dirname(os.path.realpath(__file__)) + "/../submissions"
    
    # Create directory if necessary
    os.makedirs(submissions_path, exist_ok=True)

    with open(os.path.join(submissions_path, name + str(datetime.datetime.now()) + ".csv"), 'w') as f:
        f.write("playlist_id,track_ids\n")
        for row in recs:
            f.write(str(row[0]) + ",")
            f.write(" ".join([str(el) for el in row[1]]))
            f.write("\n")