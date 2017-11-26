from sklearn import tree

#0-apple ,1-orange
#0-smooth,1-bumpy

features=[[140,0],[130,0],[150,1],[170,1]]
labels=[0,0,1,1]

clf=tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)

# give input [100,0]
#output should be apple
print clf.predict([[100,0]])

