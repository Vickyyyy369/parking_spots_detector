import * as tf from '@tensorflow/tfjs';
		const MODEL_URL = '.web_model/model.json';
		const class_dictionary=['empty','occupied'];
		async function fun(){
			const model = await tf.loadGraphModel(MODEL_URL);
			const imageInput = document.getElementById('scene');
			//imageInput.value = imageVectorToText(）; //应该还有imageSize属性，确定图片尺寸
			const imageRead = tf.browser.fromPixels(imageInput); //将HTML元素转换成3D张量
			const imageTensors=[];
			imageTensors.push(imageRead);
			const images = tf.stack(imageTensors);  
			//const batchedImage = images.expandDims(0);  //增加维度为4D
			const b=tf.scalar(255);
			const imgs=images.toFloat().div(b);   //除以255
            const predictions = model.predict(imgs);
            const predictedClass = predictions.as1D().argMax();
            const classId = (await predictedClass.data())[0];
            const result = class_dictionary[classId];
            result.print();

		}
        fun();