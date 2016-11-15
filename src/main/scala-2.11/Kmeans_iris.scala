import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans
/**
  * Created by user on 2016/11/14.
  */
object Kmeans_iris {
  def main(args: Array[String]) {
    val spark = SparkSession.builder().master("local").appName("RandomForest").getOrCreate()

    val iris = spark.read.option("header","true").option("inferSchema","true").csv("/home/user/mllib_data/iris.csv")

    val header = iris.columns.filter(_!="Species")

    val assembler = new VectorAssembler().setInputCols(header).setOutputCol("features")
    val transformed = assembler.transform(iris)

    val kmeans = new KMeans().setK(3)
    val kmModel3 = kmeans.fit(transformed)

    println("Cluster Centers: ")
    kmModel3.clusterCenters.foreach(println)

    val iris2 = assembler.transform(iris)
    val WSSSE = kmModel3.computeCost(transformed)
    println(s"Within Set Sum of Squared Errors = $WSSSE")


  }
}
