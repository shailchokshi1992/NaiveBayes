import os
import time
import urllib
import tarfile
from functools import partial

from skll.metrics import kappa
from nltk.corpus import stopwords

from sklearn.metrics import classification_report, accuracy_score
from sklearn.cross_validation import train_test_split
import NaiveBayes


print("> Start donwload 20 NewsGroup data")
DATASET_URL = "http://www.cs.cmu.edu/afs/cs/project/theo-11/www/naive-bayes/20_newsgroups.tar.gz"
BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASEDIR, "data")

archive_name = DATASET_URL.split('/')[-1]
archive_path = os.path.join(DATA_DIR, archive_name)
path_to_data = os.path.join(DATA_DIR, "20_newsgroups")

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

if not os.path.exists(archive_path):
    data_archive = urllib.URLopener()
    data_archive.retrieve(DATASET_URL, archive_path)

if not os.path.exists(path_to_data):
    with tarfile.open(archive_path, "r:gz") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=DATA_DIR)


def prepare_file(filename, datafolder=path_to_data):
    filepath = os.path.join(datafolder, filename)
    with open(filepath, 'r') as f:
        return f.read()


def get_texts(categories):
    documents = []
    classes = []

    for i, category in enumerate(categories):
        category_files_path = os.path.join(path_to_data, category)
        text_ids = os.listdir(category_files_path)
        prepare_category_file = partial(prepare_file, datafolder=category_files_path)
        texts = [prepare_category_file(x) for x in text_ids]
        documents += texts
        classes += [category] * len(texts)

    return documents, classes

print("> Read files...")
start_time = time.time()
categories = os.listdir(path_to_data)

# Get data
print("> Split data to test and train")
documents, classes = get_texts(categories)
train_docs, test_docs, train_classes, test_classes = train_test_split(documents, classes, train_size=0.5)



def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)


print(">>>>*** Train classifier ****<<<<")

# classifier = NaiveBayesTextClassifier(
#     categories=categories,
#     min_df=1,
#     lowercase=True,
#     # 127 English stop words
#     stop_words=stopwords.words('english')
# )
# classifier.train(train_docs, train_classes)
#
# print("-" * 42)
# print("{:<25}: {:>6} articles".format("Total", len(train_docs)))
# print("{:<25}: {:>6} words".format(
#     "Number of words", classifier.bag.shape[1]
# ))
# print("{:<25}: {:>6.2f} seconds".format(
#     "Parse time", time.time() - start_time
# ))
# print("-" * 42)
#
# start_time = time.time()
# print("> Start classify test data")
# predicted_classes = classifier.classify(test_docs)
# end_time = time.time()
#
#
# def category_to_number(classes, category_type):
#     return list(map(category_type.index, classes))
#
#
# print(classification_report(test_classes, predicted_classes))
# print('-' * 42)
# print("{:<25}: {:>6.2f} seconds".format(
#     "Computation time", end_time - start_time
# ))
# print("{:<25}: {:>6} articles".format("Test data size", len(test_classes)))
# print("{:<25}: {:>6.2f} %".format(
#     "Accuracy", 100 * accuracy_score(test_classes, predicted_classes))
# )
# print("{:<25}: {:>6.2f} %".format(
#     "Kappa statistics", 100 * kappa(
#         category_to_number(test_classes, categories),
#         category_to_number(predicted_classes, categories)
#     )
# ))
# print('-' * 42)
