from feature_extract import (
    read_cifar10_to_tensor,
    compute_vertical_gradient,
    compute_gradiant,
)

from svm import (
    calcSiftFeature,
    calcFeatVec,
    learnVocabulary,
    SVM_Train,
    cal_vec,
    build_center,
)

train_images, train_labels, test_images, test_labels = read_cifar10_to_tensor(
    "data/cifar-10-batches-py"
)


train_data, train_label = compute_vertical_gradient(train_images)
test_data, test_label = compute_vertical_gradient(test_images)

_, idx_void = build_center(train_images)
train_images = train_images[~idx_void]
train_labels = train_labels[~idx_void]

# data_vec, labels = cal_vec(train_images)
