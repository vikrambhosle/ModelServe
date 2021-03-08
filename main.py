from Tools.scripts.dutree import display
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, Tokenizer, HashingTF, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from IPython.display import display
import mleap.pyspark
from mleap.pyspark.spark_support import SimpleSparkSerializer

from pyspark import SparkContext, SQLContext

from flask import Flask, render_template

#  Creating a model . This

sc= SparkContext()
sqlContext = SQLContext(sc)
house_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('D:\\boston_housing.csv')
house_df.take(1)
house_df.cache()
house_df.printSchema()
house_df.describe().toPandas().transpose()

vectorAssembler = VectorAssembler(inputCols = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat'], outputCol = 'features')
vhouse_df = vectorAssembler.transform(house_df)
vhouse_df = vhouse_df.select(['features', 'medv'])
vhouse_df.show(3)

splits = vhouse_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]



lr = LinearRegression(featuresCol = 'features', labelCol='medv', maxIter=10, regParam=0.3, elasticNetParam=0.8)
pipeline = Pipeline(stages=[lr])
pipe = pipeline.fit(train_df)
sparkTransformed = pipe.transform(train_df)

#print("Coefficients: " + str(lr_model.coefficients))
#print("Intercept: " + str(lr_model.intercept))

pipe.serializeToBundle("jar:file:/tmp/mleap_python_model_export/20news_pipeline-json.zip", sparkTransformed)


# Flask im0plementation

app = Flask(__name__)
@app.route('/', methods=['GET'])
def index_page_landing():
    return render_template('index.html')
if __name__ == "__main__":
    app.run()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
