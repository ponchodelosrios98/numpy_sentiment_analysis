from collections import Counter
import numpy as np

##
## Dataset
##
print('---')
print('1. DATASET')
print('---')

# Prints Formatted Review and Label
def pretty_print_review_and_label(i):
    print(labels[i] + " : " + reviews[i][:80] + "...")

# Open Reviews Files and separate them by lines
g = open('reviews.txt','r')
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

# Open Labels (Targets) and separate them by lines
g = open('labels.txt','r')
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()

# Print Sample Labels and Reviews
def print_demo():
  print("labels.txt : reviews.txt\n")
  pretty_print_review_and_label(2137)
  pretty_print_review_and_label(12816)
  pretty_print_review_and_label(6267)
  pretty_print_review_and_label(21934)
  pretty_print_review_and_label(5297)
  pretty_print_review_and_label(4998)

print_demo()

##
## Theory Validation
##
print('---')
print('2. THEORY VALIDATION')
print('---')

# Create three Counter objects to store positive, negative and total counts
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

# Loops over all the words in all the reviews and increment the counts in the appropriate counter objects
for index in range(len(reviews)):
  # Split the review in words and add it to label counter
  wordReviews = reviews[index].split(' ')

  if labels[index] == 'POSITIVE':
    for word in wordReviews:
      positive_counts[word] += 1
      total_counts[word] += 1
  elif labels[index] == 'NEGATIVE':
    for word in wordReviews:
      negative_counts[word] += 1
      total_counts[word] += 1
  
# Prints Positive Most Common
# print(positive_counts.most_common())

# Prints Negative Most Common
# print(negative_counts.most_common())

##
## DATASET RATIOS
##
print('---')
print('3. DATASET RATIOS')
print('---')

# Create Counter object to store positive/negative ratios
pos_neg_ratios = Counter()

# Loops in every word of positive counts
for word in positive_counts:
  # Divides the number of appearances between positive and negative ratios
  # This helps to identify which are really repeated (The ones near zero)
  pos_neg_ratios[word] = positive_counts[word] / float(negative_counts[word]+1)
  # We convert it to logarithm to center all the values around netural
  # So the absolute value from neutral of the postive-to-negative ratio for a word
  # Would indicate how much sentiment (positive or negative) that word conveys.
  pos_neg_ratios[word] = np.log(pos_neg_ratios[word])

# Different Ratios
print('Ratios')
print("Ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

# print('Example of Ratio Most Common Values')
# print('Negative')
# print(list(reversed(pos_neg_ratios.most_common()))[0:10])
# print('Positive')
# print(list(pos_neg_ratios.most_common())[0:10])

##
## INPUT / OUTPUT TO NUMBERS
##
print('---')
print('4. INPUT / OUTPUT TO NUMBERS')
print('---')

# Create set named "vocab" containing all of the words from all of the reviews
vocab = set(total_counts.keys())

# Create layer_0 matrix with dimensions 1 by vocab_size, initially filled with zeros
layer_0 = np.zeros((1, len(vocab)))

# layer_0 contains one entry for every word in the vocabulary,
# We need to make sure we know the index of each word,
# We run the following function to create a lookup table that stores the index of every word.
word2index = {}

for i, word in enumerate(vocab):
  word2index[word] = i

# Counts how many times each word is used in the given review, and then store
# those counts at the appropriate indices inside `layer_0`.
def update_input_layer(review):
  """
    Modify the global layer_0 to represent the vector form of review.
    The element at a given index of layer_0 should represent
    how many times the given word occurs in the review.
    Args:
      review(string) - the string of the review
    Returns:
      None
  """
  global layer_0
  layer_0 *= 0

  for word in review.split(' '):
    layer_0[0][word2index[word]] += 1

# Updates Layer_0 Values with first review
update_input_layer(reviews[0])
print('Layer_0 Updated by first review')
print(layer_0[0])

# Displays Label Positive = 1 or Negative = 0
def get_target_for_label(label):
  if label == 'POSITIVE':
    return 1
  return 0