from confusion_matrixOne import ConfusionMatrix 
labels = ["cat", "dog", "velociraptor", "kraken", "pony"]
confusionMatrix = ConfusionMatrix(labels)

confusionMatrix.update("cat", "cat")
confusionMatrix.update("cat", "dog")
confusionMatrix.update("kraken", "velociraptor")
confusionMatrix.update("velociraptor", "velociraptor")

confusionMatrix.plot()