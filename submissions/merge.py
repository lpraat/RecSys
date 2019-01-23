playlists = []

l1 = []
l2 = []

file1 = 'first.csv'
file2 = 'second.csv'

with open(file1) as f:
    lines = f.readlines()[1:]

    for line in lines:
        if line == '\n':
            continue
        playlist, tracks = line.split(",")
        playlist = int(playlist)
        tracks = [int(x) for x in tracks.split(" ")]
        l1.append(tracks)
        playlists.append(playlist)

with open(file2) as f:
    lines = f.readlines()[1:]

    for line in lines:
        if line == '\n':
            continue
        playlist, tracks = line.split(",")
        playlist = int(playlist)
        tracks = [int(x) for x in tracks.split(" ")]
        l2.append(tracks)

final_l = []


def merge(l1, l2):
    l3 = []
    curr = l1
    while len(l3) < 10:

        if curr[0] not in l3:
            el = curr.pop(0)
            l3.append(el)
            curr = l1 if curr == l2 else l2
        else:
            curr.pop(0)

    return l3


with open('new.csv', 'w') as f:
    f.write("playlist_id,track_ids")
    for i, p in enumerate(playlists):
        f.write("\n")
        f.write(str(p) + ",")
        f.write(" ".join([str(x) for x in merge(l1[i], l2[i])]))
