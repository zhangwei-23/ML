# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf_vec = TfidfVectorizer()
#
# documents = [
#     'this is the bayes document',
#     'this is the second second document',
#     'and the third one',
#     'is this the document'
# ]
# tfidf_matrix = tfidf_vec.fit_transform(documents)
#
# print('不重复的词:', tfidf_vec.get_feature_names())
# print('每个单词的ID:', tfidf_vec.vocabulary_)
# print('每个单词的tfidf值:', tfidf_matrix.toarray())

# --------------------------------------------------------------------------------------------------

#英文文档分词
# import nltk
# word_list = nltk.word_tokenize(text) #分词
# nltk.pos_tag(word_list) #标注单词的词性

#中文文档分词
# import jieba
# word_list = jieba.cut (text) #中文分词


import os
import jieba

def cut_word(file_path):
    '''
    对文本进行切词
    :param file_path:txt文本路径
    :return:用空格分词的字符串
    '''
    text = ''
    text = open(file_path,'r',encoding='gb18030').read()  #r 以只读方式打开文件
    word_list = jieba.cut(text)
    for word in word_list:
        word_with_spaces += word + ''
    return text_with_spaces

def loadfile(file_dir,label):
    '''
    :param file_dir: 保存txt文件
    :param label: 文档标签
    :return: 分词后的文档列表和标签
    '''
    file_list = os.listdir(file_dir)
    word_list = []
    labels_list = []
    for file in file_list:
        file_path = file_dir + '/' + file
        word_list.append(cut_word(file_path))
        labels_list.append(label)
    return word_list, labels_list

# 训练数据
train_words_list1, train_labels1 = loadfile('D:/Learning/ML & DL/ML/Bayes/text_classification-master/text classification/train/女性.txt', '女性')
train_words_list2, train_labels2= loadfile('D:\Learning\ML & DL\ML\Bayes\text_classification-master\text classification/train/体育', '体育')
train_words_list3, train_labels3= loadfile('D:\Learning\ML & DL\ML\Bayes\text_classification-master\text classification/train/文学', '文学')
train_words_list4, train_labels4= loadfile('D:\Learning\ML & DL\ML\Bayes\text_classification-master\text classification/train/校园', '校园')

train_words_list = train_words_list1 + train_words_list2 + train_words_list3 + train_words_list4
train_labels = train_labels1 + train_labels2 + train_labels3 + train_labels4

# 测试数据
test_words_list1, test_labels1 = loadfile('D:\Learning\ML & DL\ML\Bayes\text_classification-master\text classification\test/女性', '女性')
test_words_list2, test_labels2 = loadfile('D:\Learning\ML & DL\ML\Bayes\text_classification-master\text classification\test/体育', '体育')
test_words_list3, test_labels3 = loadfile('D:\Learning\ML & DL\ML\Bayes\text_classification-master\text classification\test/文学', '文学')
test_words_list4, test_labels4 = loadfile('D:\Learning\ML & DL\ML\Bayes\text_classification-master\text classification\test/校园', '校园')

test_words_list = test_words_list1 + test_words_list2 + test_words_list3 + test_words_list4
test_labels = test_labels1 + test_labels2 + test_labels3 + test_labels4

# 加载停用词
stop_words = open('D:\Learning\ML & DL\ML\Bayes\text_classification-master\stop/stopword.txt', 'r', encoding='utf-8').read()
stop_words = stop_words.encode('utf-8').decode('utf-8-sig') # 列表头部\ufeff处理
stop_words = stop_words.split('\n') # 根据分隔符分隔

# 计算单词权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)

train_features = tf.fit_transform(train_words_list)
# 上面fit过了，这里transform
test_features = tf.fit_transform(test_words_list)

# 多项式贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
predicted_labels = clf.predict(test_features)

# 得到测试集的特征矩阵
# test_tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5, vocabulary=train_vocabulary)
# test_features=test_tf.fit_transform(test_contents)
# predicted_labels=clf.predict(test_features)

# 计算准确率
print('准确率：', metrics.accuracy_score(test_labels,predicted_labels))