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
