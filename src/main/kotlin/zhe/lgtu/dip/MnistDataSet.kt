package zhe.lgtu.dip

import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import java.io.File
import java.util.*

class MnistDataSet(seed: Long, basePath: String, height: Long,width: Long, channels: Long, batchSize: Int, outputNum: Int) {

    val trainIter:RecordReaderDataSetIterator

    init {
        val randNumGen = Random(seed)
        val trainData = File(basePath)
        val trainSplit = FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
        val labelMaker = ParentPathLabelGenerator()
        val trainRR = ImageRecordReader(height, width,channels, labelMaker)
        trainRR.initialize(trainSplit)
        trainIter = RecordReaderDataSetIterator(trainRR, batchSize, 1,outputNum)
        // pixel values from 0-255 to 0-1 (min-max scaling)
        val scaler = ImagePreProcessingScaler(0.0, 1.0)
        scaler.fit(trainIter)
        trainIter.preProcessor = scaler
    }

    fun getTestAndTrain(): Pair<MutableList<DataSet>, DataSet> {

        val trainSetList : MutableList<DataSet> = mutableListOf()
        val testSetList : MutableList<DataSet> = mutableListOf()
        while (trainIter.hasNext()) {
            val allData = trainIter.next()

            allData.shuffle()
            val testAndTrain = allData.splitTestAndTrain(0.8)
            trainSetList.add(testAndTrain.train)
            testSetList.add(testAndTrain.test)
        }

        val testSet = DataSet.merge(testSetList)

        return Pair(trainSetList , testSet)
    }
}

