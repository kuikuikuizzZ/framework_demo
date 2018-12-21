

ctpn笔记

3.1 detecting text in fine scale proposal

	这里主要借鉴了faster r-cnn 的rpn(region proposal network)的概念，同时又指出了直接使用rpn 在文本行 detect的位置有一定的缺陷，一个是使用 rpn检测物体的时候一般会有明显的轮廓，而文本行这样的特征并不明显；另一个是文本行一般都具有序列的特征，直接使用 rpn 有所不妥。

						图1 ：使用 rpn 与使用 ctpn 的对比

	由此，文章提出了使用固定水平位置的回归anchor，其原理如下：首先 VGG16在 conv5 的 feature map 在 input 的步长和感受野为16和228pixels。detector 会扫描conv5 上的每一个 pixels，相当于以步长为16的窗口在 input上滑动。然后在每一个位置上，就是 conv 的每一个 pixel 上有k 个 锚点(anchor)对应。每一个锚点对应不同的文本小框的高度h_y，由于是在同一个 pixel 下的所以具有共同的水平位置。文章这里使用了（273->11）(0.7的比值递减)。然后我们可以计算预测的文本小框与 ground true 小框的坐标参数了

	预测值： v_c = (c_y-c_{y}^a)/h^a     v_h = log(h/h^a)

	ground true:  v_{c}^* = (c_{y}^*-c_{y}^a)/h^a     v_{h}^* = log(h^*/h^a)

 这里v_c, v_{c}^* 为 bounding box 的中心位置(y轴)， v_h, v_{h}^*  为 bouding box的高度， c_{y}^a ，h^a 

是锚点的中心位置(y轴)和高度（就是（273->11）参考值）这些都可以在训练之前预先计算出来。每一次的预测相当于bounding box的水平高度和 anchor 的位置参数( c_{y}^a ，h^a) 都是固定的。

然后模型的输出就是在每一个 conv5 location上的text/non-text 的分数以及每一个 anchor 的位置参数(v_c, v_{h} )。



3.2 recurrent connectionist text proposals

一般的文本行提取都会有 connect components的步骤，文章使用 rnn 完成了这一步。

这里是使用rnn的常用方法了，lstm+Bidirectional，不过使用的时候需要 reshape 一下，因为使用 rnn 默认扔进去的格式是(batch_size, time_steps, channels) 所以把纵向的维度和 batch的维度合并输进去

    ## 首先 reshape 成 b[0]*b[1],b[2],b[3]
    def reshape(x):
        b = tf.shape(x)
        x = tf.reshape(x,[b[0]*b[1],b[2],b[3]])
        return x
    ## output reshape 为原来的 feature map 的形状，不过 channel数有所不同
    def reshape2(x):
        x1,x2 = x
        b = tf.shape(x2)
        x = tf.reshape(x1,[b[0],b[1],b[2],256])
        return x 
    



							图2 ：non-rnn/rnn 比较

3.3 Side-refinement

	前面一直谈的都是宽度为16 的bounding box，不过通常来说我们text detection都是指提取出一整个文本行，所以怎么把这些小框连起来呢。原则首先 text/non-text score > 0.7，然后我们对不同的候选框 B_i 跟它的近邻候选框 B_j  筛选: (i)  B_i ， B_j 需要在水平距离上是最近的，(ii)  B_i ， B_j的水平距离不超过50，(iii) 他们的竖直方向上 overlap > 0.7 

 	另外，我们有时候获得的候选框的水平位置不一定是最优的，因为我们的 stride 是16pixel，所以按照上面的回归方法有可能回归到跟 ground_true 有几个 pixel 的差距。由此，文章引入了一个修正x位置参数的修正量。

	预测值 o = (x_{side} - c_{x}^a)/w^a，  ground true: o^* = (x_{side}^* - c_{x}^a)/w^a 

x_{side} 表示bounding box 左或右一边的坐标， w^a 表示bounding box 的宽度，这里是固定值16。 c_{x}^a 是指anchor 的中心位置。不过这里的预测值x_{side} 是指连接后的bounding box的边的位置，就是整个文本行的时候，不是在每一个小框。



3.4 Model Outputs and Loss Functions



									图3 ctpn 原理

输出主要有三个一个是每个anchor的text/non-text的分数，一个是每一个anchor 的竖直位置信息，一个是每个anchor的side-refinement，loss 如下定义：



这里各个N 表示计算相应loss 的时候的总数，L^{cl} 是指所有的anchor ,L^{re}_v 是指text score>0.7的anchor ,L^{re}_v  是指划成同一个文本行的所有anchor。在训练的时候，也需要给出每一个bounding box 对于左右边的距离o^*，如果预测的距离o不对就需要惩罚。



3.5 Training and Implementation Details

文章使用的是SGD优化，其中momentum为0.9，weight decay 为0.0005，开始的learning rate 为0.001，后续的learning rate 为0.0001

训练的labels。训练的时候，我们首先会给出每一个anchor有没有text的label，这个也是预先给出的，判断规则：(i) 只要该anchor与ground true IoU > 0.7 (ii) 对于某些ground true boxes，所有anchor 的 IoU 都不满足 > 0.7的，选择IoU最大的那个。所以即使只有很小的text pattern 也可能被看作是positive anchor，negative anchor 定义为IoU < 0.5 的anchor。





代码分析

    # 该函数主要是在feature_map上逐个pixel移动的时候，在原图上是怎么移动，返回原图上anchor的坐标,返回的shape为(feature_map_pixels,(10,4))
    def gen_anchor(featuresize,scale) #feature size 表示最后一层的feature map size
    
    ## 输入的是两组boxes,输出是这两组boxes的overlap值,shape为(anchor_nums,gtbox_nums)
    ## 用到了cal_iou
    def cal_overlaps():
        #calculate the intersection of  boxes1(anchor) and boxes2(GT box)
        for i in range(boxes1.shape[0]):
            overlaps[i][:] = cal_iou(boxes1[i],area1[i],boxes2,area2)
           
    ## 输入两个boxes 的(x1,y1,x2,y2)和面积，输出IoU的值
    def cal_iou(box1,box1_area, boxes2,boxes2_area):
        
    ## 该函数输入为(原图的长宽，featuremap的长宽，缩小的尺寸，
    ## ground true boxes的(x1,y1,x2,y2))
    def cal_rpn(imgsize,featuresize,scale,gtboxes)
    	# 首先给出这个图片的anchor位置信息
        base_anchor = gen_anchor(featuresize,scale)
        #calculate anchor与ground true 的 iou
        overlaps = cal_overlaps(base_anchor,gtboxes)
        #init labels -1 don't care  0 is negative  1 is positive
        labels = np.empty(base_anchor.shape[0])
        labels.fill(-1) ## don't care
        #for each GT box corresponds to an anchor which has highest IOU 
        # 对于每一个 GT box 找到对应最大的IOU anchor
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        
        #the anchor with the highest IOU overlap with a GT box
        # 对于每一个 anchor 找到最大的GT box
        anchor_argmax_overlaps = overlaps.argmax(axis=1)
        # 每一个 anchor 找到最大的GT box 对应的值
        anchor_max_overlaps = overlaps[range(overlaps.shape[0]),
                                       anchor_argmax_overlaps] 
        
        #IOU > IOU_POSITIVE，>0.7 
        labels[anchor_max_overlaps>IOU_POSITIVE]=1
        #IOU <IOU_NEGATIVE  <0.5
        labels[anchor_max_overlaps<IOU_NEGATIVE]=0
        #ensure that every GT box has at least one positive RPN region
        labels[gt_argmax_overlaps] = 1
    	
    	##这保留在原图上的anchor 
        #only keep anchors inside the image
        outside_anchor = np.where(
           (base_anchor[:,0]<0) |
           (base_anchor[:,1]<0) |
           (base_anchor[:,2]>=imgw)|
           (base_anchor[:,3]>=imgh)
           )[0]
        labels[outside_anchor]=-1
    
        ### 控制positive labels 和 negative labels 的数量
        #subsample positive labels ,if greater than RPN_POSITIVE_NUM(default 128)
        fg_index = np.where(labels==1)[0]
        if(len(fg_index)>RPN_POSITIVE_NUM):
            labels[np.random.choice(fg_index,
                                    len(fg_index)-RPN_POSITIVE_NUM,
                                    replace=False)]=-1
    
        #subsample negative labels 
        bg_index = np.where(labels==0)[0]
        num_bg = RPN_TOTAL_NUM - np.sum(labels==1)
        if(len(bg_index)>num_bg):
            #print('bgindex:',len(bg_index),'num_bg',num_bg)
            labels[np.random.choice(bg_index,
                                    len(bg_index)-num_bg,
                                    replace=False)]=-1
        #calculate bbox targets 返回的是(Vc,Vh)每个 anchor 的 log 变换后的参数
        bbox_targets = bbox_transfrom(base_anchor,gtboxes[anchor_argmax_overlaps,:])
        # 返回的是每个 anchor 的 labels,第二个返回的是text=1的 anchor 的位置参数，和 base_anchor 的位置
        return [labels,bbox_targets],base_anchor
    
    # 完成log(cy-cya)/ha的转换
    def bbox_transfrom() 
    
    # 这个是从 anchor 的坐标信息和优化后的(Vc,Vh)的值反推出 bboxes 的中心坐标(Cx,Cy)和长宽(16,h)
    def bbox_transfor_inv(anchor,regr):
    
    # 产生训练的样本，label已经处理好了
    def gen_sample(xmlpath,imgpath,batch_size=1)




