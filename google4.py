from sklearn.datasets import load_iris
iris=load_iris()

x=iris.data
y=iris.target

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)

#knearest neighbors classifier
#from sklearn.neighbors import KNeighborsClassifier
#clf=KNeighborsClassifier()

from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)

predictions=clf.predict(x_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test,predictions)

