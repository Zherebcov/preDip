import java.io.File
import java.util.HashMap
import java.util.Random

import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.BaseImageRecordReader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import zhe.lgtu.dip.DataUtilities
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.MapSchedule
import org.nd4j.linalg.schedule.ScheduleType
import org.slf4j.LoggerFactory
import kotlin.reflect.KFunction0
import kotlin.reflect.KFunction1

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

    val log = LoggerFactory.getLogger(MyMnist::class.java)
    val basePath = "D:\\Jktu\\Datasets\\symbols"
    val height = 25
    val width = 20
    val channels = 1 // single channel for grayscale images
    val outputNum = 22 // 10 digits classification
    val batchSize = 54
    val nEpochs = 5
    val seed = 1234


    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {


        val (trainIter, testIter) = NewTimed("Create DataSet for", MyMnist::CreateDataSet)
        val net = NewTimed("Create Network for", MyMnist::CreateNetwork)

        // evaluation while training (the score should go down)
        (0 until nEpochs).forEach {
            NewTimed("Completed epoch {} ms") { net.fit(trainIter) }
            NewTimed("Completed evaluation {} ms") {
                val eval = net.evaluate(testIter)
                log.info(eval.stats(false)+'\n'+eval.confusionMatrix())
            }
            trainIter.reset()
            trainIter.labels
            testIter.reset()
        }

        NewTimed("Save model"){ModelSerializer.
                writeModel(net, File("$basePath/minist-model.zip"), true)}
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
    inline fun CreateDataSet(): Pair<RecordReaderDataSetIterator, RecordReaderDataSetIterator> {
        val randNumGen = Random(seed.toLong())

        // vectorization of train data
        val trainData = File("$basePath")
        val trainSplit = FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
        val labelMaker = ParentPathLabelGenerator() // parent path as the image label
        val trainRR = ImageRecordReader(height.toLong(), width.toLong(), channels.toLong(), labelMaker)
        trainRR.initialize(trainSplit)
        val trainIter = RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum)
        // pixel values from 0-255 to 0-1 (min-max scaling)
        val scaler = ImagePreProcessingScaler(0.0, 1.0)
        scaler.fit(trainIter)
        trainIter.preProcessor = scaler

        // vectorization of test data
        val testData = File("$basePath")
        val testSplit = FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
        val testRR = ImageRecordReader(height.toLong(), width.toLong(), channels.toLong(), labelMaker)
        testRR.initialize(testSplit)
        val testIter = RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum)

        testIter.preProcessor = scaler // same normalization for better results
        return Pair(trainIter, testIter)
    }

    @JvmStatic
    inline fun CreateNetwork(): MultiLayerNetwork {
        log.info("Network configuration and training...")
        val lrSchedule = HashMap<Int, Double>()
        lrSchedule[0] = 0.01 // iteration #, learning rate
        lrSchedule[600] = 0.005
        lrSchedule[1000] = 0.001

        val conf = NeuralNetConfiguration.Builder()
                .seed(seed.toLong())
                .l2(0.0005)
                .updater(Nesterovs(MapSchedule(ScheduleType.ITERATION, lrSchedule)))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1) // nIn need not specified in later layers
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height.toLong(), width.toLong(), channels.toLong())) // InputType.convolutional for normal image
                .backprop(true).pretrain(false).build()

        var net = MultiLayerNetwork(conf)
        net.init()
        net.setListeners(ScoreIterationListener(20000 / batchSize / 20))
        log.debug("Total num of params: {}", net.numParams())
        return net
    }

    inline
    private fun NewTimed(text: String, function: () -> Unit) {
        var start = System.currentTimeMillis()
        function()
        var timeConsumedMillis = System.currentTimeMillis() - start
        log.info("$text {} ms", timeConsumedMillis)
    }
}
