from spleeter.separator import Separator

def separate_vocals(songs_dir, filenames, save_directory):
    separator = Separator("spleeter:2stems")
    for filename in filenames:
        separator.separate_to_file(songs_dir + filename + ".wav", save_directory, filename_format="{filename}_{instrument}.{codec}")