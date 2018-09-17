package zhe.lgtu.dip

import java.io.File
import java.util.HashMap

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.MapSchedule
import org.nd4j.linalg.schedule.ScheduleType
import org.slf4j.LoggerFactory
import kotlin.reflect.KFunction0

/**
 * Handwritten digits image classification on MNIST dataset (99% accuracy).
 * This example will download 15 Mb of data on the first run.
 * Supervised learning best modeled by CNN.
 *
 * @author hanlon
 * @author agibsonccc
 * @author fvaleri
 */
object MyMnist {

    val log = LoggerFactory.getLogger(this::class.java)
    val basePath = System.getenv("SYMBOL")
    val height = 10
    val width = 10
    val channels = 1 // single channel for grayscale images
    val outputNum = 21 // 10 digits classification
    val batchSize = 108
    val nEpochs = 30
    val seed = 1234
    val convLay = Pair(3,3)

    val timeLearn = listOf<Int>()
    val timeEval = listOf<Int>()


    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {

        val mnistDataSet = MnistDataSet(seed.toLong(),basePath,height.toLong(), width.toLong(), channels.toLong(), batchSize, outputNum)
        val (trainIter, testIter) = mnistDataSet.getTestAndTrain()
        val net = CreateNetwork(seed.toLong(),channels.toLong(),outputNum,height.toLong(), width.toLong(),convLay).net
        Fit(trainIter, testIter, net)


    }

    @JvmStatic
    inline fun <T> NewTimed(text: String, build: KFunction0<T>): T {

        var start = System.currentTimeMillis()
        val set = build.invoke()
        var timeConsumedMillis = System.currentTimeMillis() - start
        log.info("$text {} ms", timeConsumedMillis)
        return set
    }

    @JvmStatic
    fun Fit(trainIter: MutableList<DataSet>, testIter: DataSet, net: MultiLayerNetwork) {
        (0 until nEpochs).forEach {
            NewTimed("Completed epoch ${it+1}") {  trainIter.forEach { net.fit(it)}}
            NewTimed("Completed evaluation ${it+1}") {

                val eval = Evaluation (outputNum)
                val output = net.output(testIter.getFeatures());
                eval.eval(testIter.getLabels(), output);
                log.info(eval.accuracy().toString() /*+ eval.confusionMatrix()*/)

            }
        }

        NewTimed("Save model") {
            ModelSerializer.writeModel(net, File("$basePath/minist-model.zip"), true)
        }
    }

    inline
    private fun NewTimed(text: String, function: () -> Unit) {
        var start = System.currentTimeMillis()
        function()
        var timeConsumedMillis = System.currentTimeMillis() - start
        log.info("$text {} ms", timeConsumedMillis)
    }
}
