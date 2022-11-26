import json
import copy
import os
import argparse as ap
from unidecode import unidecode
from preprocessing import separate_vocals
from wrapper import align, preprocess_from_file

def get_args_list():
    parser = ap.ArgumentParser()
    parser.add_argument("-i", "--input", nargs=1, default=["/data"])
    parser.add_argument("-o", "--output", nargs=1, default=["/result"])
    return parser.parse_args()

def get_json_timestamp(word_align, json_in):
    json_out = copy.deepcopy(json_in)
    resolution = 256 / 22050 * 3000
    idx = 0
    for i in range(len(json_out)):
        for lrcs in json_out[i]["l"]:
            lrcs["s"] = round(word_align[idx][0] * resolution)
            lrcs["e"] = round(word_align[idx][1] * resolution)
            idx += 1
        json_out[i]["s"] = json_out[i]["l"][0]["s"]
        json_out[i]["e"] = json_out[i]["l"][-1]["e"]
    return json_out

def align_wav_input(json_lyrics, wav_file):
    lyrics = unidecode(" ".join(lrcs["d"] for seg in json_lyrics for lrcs in seg["l"]))
    audio, words, lyrics_p, idx_word_p, idx_line_p = preprocess_from_file(audio_file=wav_file, lyrics=lyrics, word_file=None)
    word_align, words = align(audio, words, lyrics_p, idx_word_p, idx_line_p, method="MTL", cuda=True)
    return get_json_timestamp(word_align, json_lyrics)

def predict(labels_dir, songs_dir, filenames, writeto="./predictions/", use_processed_audio=False):
    l = len(filenames)
    for i in range(l):
        json_lyrics = json.load(open(labels_dir + filenames[i] + ".json"))
        wav_file = songs_dir + filenames[i] + ("_vocals.wav" if use_processed_audio else ".wav")
        print(f"\n==== [{i+1}/{l}]: {wav_file}")
        try:
            res = align_wav_input(json_lyrics, wav_file)
            with open(writeto + filenames[i] + ".json", "w", encoding="utf8") as fout:
                json.dump(res, fout, ensure_ascii=False)
        except:
            print("==== [!] Error with alignment")
            with open(writeto + filenames[i] + ".json", "w", encoding="utf8") as fout:
                json.dump(json_lyrics, fout, ensure_ascii=False)

def get_filenames(dir):
    filenames = [os.path.splitext(i)[0] for i in os.listdir(os.path.join(dir))]
    return filenames

if __name__ == "__main__":
    # labels_dir = "/data/lyrics/"
    # songs_dir = "/data/songs/"
    # results_dir = "/result/"

    args = get_args_list()
    labels_dir = args.input[0] + "/lyrics/"
    songs_dir = args.input[0] + "/songs/"
    results_dir = args.output[0] + "/"

    if not os.path.exists(results_dir):
        os.system(f"sudo mkdir {results_dir}")
    filenames = get_filenames(labels_dir)

    # preprocess audio
    preprocessed_dir = "./processed_audio/"
    separate_vocals(songs_dir, filenames, preprocessed_dir)

    # predict
    predict(labels_dir, preprocessed_dir, filenames, results_dir, True)