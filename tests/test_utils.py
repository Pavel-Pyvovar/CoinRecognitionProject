import re

from util.utils import sift_is_descriptor, ROOT_SIFT

if __name__ == '__main__':
    re.match("(SIFT)_*")
    print(sift_is_descriptor(ROOT_SIFT))
    print(sift_is_descriptor(f"{ROOT_SIFT}(nfeatures=1000)"))