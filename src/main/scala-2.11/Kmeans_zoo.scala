import org.apache.spark.sql.SparkSession

/**
  * Created by user on 2016/11/14.
  */
object Kmeans_zoo {
  def main(args: Array[String]) {

    val spark = SparkSession.builder().master("local").appName("RandomForest").getOrCreate()
    val zoo = spark.read.option("inferSchema","true").csv("/home/user/mllib_data/zoo.data")
    val header = zoo.columns.filter(_!="_c0")

    import org.apache.spark.ml.feature.VectorAssembler
    val assembler = new VectorAssembler().setInputCols(header).setOutputCol("features")

    val transformed = assembler.transform(zoo)

    import org.apache.spark.ml.clustering.KMeans
    val kmeans = new KMeans().setK(7)

    val model = kmeans.fit(transformed)

    model.clusterCenters.foreach(println)

    val result = model.transform(transformed)
    val WSSSE = model.computeCost(transformed)
    println(s"Within Set Sum of Squared Errors = $WSSSE")


    result.select("_c0","prediction").show


  }
}
