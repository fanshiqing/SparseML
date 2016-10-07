package org.apache.spark.mllib.sparselr

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap
import org.apache.spark.mllib.sparselr.Utils._
import org.apache.spark.SparkEnv
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast

import scala.collection.Map

object LogisticRegression {
    def train(input: RDD[(Array[Double], Matrix)],
              optimizer: Optimizer
              ): (Array[Int], Array[Double]) = {

      val hdfsIndex2global = new Int2IntOpenHashMap()
      var index = 0

//      input.map { point =>
//        point._2 match {
//          case x: CompressedSparseMatrix =>
//            println("x.length: " + x.mappings.length)
//          case _ =>
//            throw new IllegalArgumentException(s"dot doesn't support ${input.getClass}.")
//        }
//      }.count

      val global2hdfsIndex = input.map { point =>
        point._2 match {
          case x: CompressedSparseMatrix =>
            x.mappings
          case _ =>
            throw new IllegalArgumentException(s"dot doesn't support ${input.getClass}.")
        }
      }.flatMap(t => t).distinct.collect()

      global2hdfsIndex.foreach{value =>
        hdfsIndex2global.put(value, index)
        index += 1
      }

      val bcHdfsIndex2global = input.context.broadcast(hdfsIndex2global)

      val examples = input.map(global2globalMapping(bcHdfsIndex2global)).cache()

      val numTraining = examples.count()
      println(s"Training: $numTraining.")

      SparkEnv.get.blockManager.removeBroadcast(bcHdfsIndex2global.id, true)

      val examplesTest = examples.mapPartitions(_.flatMap {
        case (y, part) => part.asInstanceOf[CompressedSparseMatrix].tupletIterator(y)})

      val weights = Vectors.dense(new Array[Double](global2hdfsIndex.size))

      val newWeights = optimizer.optimize(examplesTest, weights)

      ((global2hdfsIndex, newWeights.toArray))
    }

  /**
    * Default option.
    *
    * @param input native featureId array RDD
    * @return A map from distinct featureId which appeared in the training dataset to global compressed new indices
    */
  def _getGlobalIndices0(input: RDD[Array[Int]]) : Map[Int, Int] = {
    input.flatMap(t => t).distinct().collect().zipWithIndex.toMap
  }

  /**
    * Directly collect the featureIds in the RDD to driver
    */
  def _getGlobalIndices1(input: RDD[Array[Int]]) : Map[Int, Int] = {
    input.collect().flatMap(t => t).distinct.zipWithIndex.toMap
  }

  /**
    * Use ArrayBuffer to cache the intermediate indices
    */
  def _getGlobalIndices2(input: RDD[Array[Int]]) : Map[Int, Int] = {
    val global2hdfsIndex = new scala.collection.mutable.ArrayBuffer[Int]()
    input.mapPartitions { iter =>
      val buffer = new collection.mutable.ArrayBuffer[Int]()
      while (iter.hasNext) {
        buffer.appendAll(iter.next())
      }
      Iterator(buffer.toArray.distinct)
    }.collect().foreach { indices =>
      global2hdfsIndex.appendAll(indices)
    }
    global2hdfsIndex.toArray.distinct.zipWithIndex.toMap
  }

  /**
    * Use HashSet to cache the intermediate indices.
    *
    * Theoretically this approach has the same time complexity with the approach above which
    * takes ArrayBuffer to cache the intermediate indices while with less memory overhead.
    */
  def _getGlobalIndices3(input: RDD[Array[Int]]) : Map[Int, Int] = {
    val global2hdfsIndex = new collection.mutable.HashSet[Int]()
    input.mapPartitions { iter =>
      val buffer = new collection.mutable.HashSet[Int]()
      while (iter.hasNext) {
        buffer ++= iter.next()
      }
      Iterator(buffer.toArray)
    }.collect().foreach { indices =>
      global2hdfsIndex ++= indices
    }
    global2hdfsIndex.toArray.zipWithIndex.toMap
  }

  /**
    * TreeAggregation with default depth to prevent returning all partial results to the driver where
    * a single pass reduce would take place as the classic aggregate does.
    */
  def _getGlobalIndices4(input: RDD[Array[Int]]) : Map[Int, Int] = {
    input.treeAggregate(Array.empty[Int])(
      (set, v) => set.union(v),
      (set1, set2) => set1.union(set2).distinct,
      2).zipWithIndex.toMap
  }

  def train2(input: RDD[(Double, Vector)],
             optimizer: Optimizer
            ): (Array[Int], Array[Double]) = {
    val global2hdfsIndex = input.map { point =>
      point._2 match {
        case x: SparseVector =>
          x.indices ++ x.binaryIndices

        case _ =>
          throw new IllegalArgumentException(s"dot doesn't support ${input.getClass}.")
      }
    }.flatMap(t => t).distinct().cache()

    // To get a Map of old sparse featureIds to new compressed featureIds for the memory savings,
    // there are many other optional approaches listed as above, i.e., _getGlobalIndices0/1/2/3/4.
    // Kindly choose the approach that best suits your applications' specific dataset.
    val hdfsIndex2global : Map[Int, Long] = global2hdfsIndex.zipWithIndex().collectAsMap()
    val bcHdfsIndex2global = input.context.broadcast(hdfsIndex2global)

    val examples = input.map(hdfs2globalMapping(bcHdfsIndex2global)).cache()

    val numTraining = examples.count()
    println(s"Training: $numTraining.")

    SparkEnv.get.blockManager.removeBroadcast(bcHdfsIndex2global.id, true)

    val weights = Vectors.dense(new Array[Double](hdfsIndex2global.size))

    val newWeights = optimizer.optimize(examples, weights)

    ((global2hdfsIndex.collect(), newWeights.toArray))
  }

  //globalId to localId for mappings in Matrix
    def global2globalMapping(bchdfsIndex2global: Broadcast[Int2IntOpenHashMap])
                     (partition: (Array[Double], Matrix)): (Array[Double], Matrix) = {
      val hdfsIndex2global = bchdfsIndex2global.value

      partition._2 match {
        case x: CompressedSparseMatrix =>
          val local2hdfsIndex = x.mappings
          for (i <- 0 until local2hdfsIndex.length) {
            local2hdfsIndex(i) = hdfsIndex2global.get(local2hdfsIndex(i))
          }
        case _ =>
          throw new IllegalArgumentException(s"dot doesn't support ${partition.getClass}.")
      }
      partition
    }

  def hdfs2globalMapping(bchdfsIndex2global: Broadcast[Map[Int, Long]])
                        (point: (Double, Vector)): ((Double, Vector)) = {
    val hdfsIndex2global = bchdfsIndex2global.value

    point._2 match {
      case x: SparseVector =>
        for (i <- 0 until x.indices.length) {
          x.indices(i) = hdfsIndex2global.get(x.indices(i)).get.toInt
        }
        for (i <- 0 until x.binaryIndices.length) {
          x.binaryIndices(i) = hdfsIndex2global.get(x.binaryIndices(i)).get.toInt
        }
      case _ =>
        throw new IllegalArgumentException(s"dot doesn't support ${point.getClass}.")
    }
    point
  }
}
