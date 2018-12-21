Clever 分布式实践

	在clever 中是默认是用kubeflow中的tfjob起tensorflow 训练服务，在训练的时候可以选择分布式进行训练，不过需要在代码中作相应的修改以适应分布式训练。我们这里以ps-woker的架构为例，在tensorflow 中这种架构比较的通用，需要修改的代码量也是比较小的，不过不一定是性能最优的（还要看很多东西，譬如集群节点的网络带宽，拓扑结构，代码的质量等）。

	首先，我们会先从把任务写成分布式，然后再解释在clever 平台上，原来的分布式代码需要做什么样的修改，就可以完成一键分布式训练。

单机训练转成分布式需要的改动

一般tensorflow的分布式都需要定义不同节点的ip地址和工作端口，然后定义谁是ps，谁是worker，具体体现在需要填写类似job_name,task_index 等一些config。下面以一个有1个ps，3个wokers 的分布式训练为例。只列出关键的代码：

    # 首先需要给不同的节点分配不同的工作
    tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
    tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
    FLAGS = tf.app.flags.FLAGS
    
    ## 这里ps-**指代的是不同节点的ip地址
    parameter_servers = ["pc-01:2222"]
    workers = [	"pc-02:2222", 
    			"pc-03:2222",
    			"pc-04:2222"]
    # 这里相当于告诉tensorflow 这几台机子作为一个集群了(不同节点是同一份集群的名单)
    cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})
    ### 不同节点负责不同工作
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    
    ## 如果调用这个程序的是参数服务器
    if FLAGS.job_name == 'ps':
            server.join()
        is_chief = (FLAGS.task_index == 0)

然后，你需要在每台机子上把分布式的程序跑起来

    pc-01$ python example.py --job_name="ps" --task_index=0 
    pc-02$ python example.py --job_name="worker" --task_index=0 
    pc-03$ python example.py --job_name="worker" --task_index=1 
    pc-04$ python example.py --job_name="worker" --task_index=2 



下面的东西就跟单机训练差不多了，我们以一个简单的分类任务为例：

    #一般tensorflow的程序结构
    with tf.device('/gpu:0'):
        global_step  = tf.Variable(0,trainable=False)
        X,y = dataset_iterator.get_next()
        # inference step
        y_logits = inference(X)
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits,
                                                             labels=y)
        loss = tf.reduce_mean(entropy)
        train_op = optimizer.minimize(loss,global_step = global_step)
        with tf.name_scope('init_and_save'):
            init = tf.global_variables_initializer()
            merged = tf.summary.merge_all()
            saver = tf.train.Saver()
            
        with tf.Session() as sess:
            init.run()
            for i in range(n_epochs):
                for step in range(n_iterations_per_epoch):
                    _,losses =sess.run([train_op,loss])



    #分布式结构，"<---" 表示需要作改动的地方
    # 1.需要告诉tensorflow,集群是哪几个节点构成，现在调用的是哪个节点
    # <---- with tf.device('/gpu:0'):
    with tf.device(tf.train.replica_device_setter(     
        cluster=cluster,
        worker_device="/job:worker/task:%d" % FLAGS.task_index)):
        # 2. global_step 分布式专用的接口，方便管理不同机子的跑的step
        # <----  global_step  = tf.Variable(0,trainable=False)
    	global_step = tf.contrib.framework.get_or_create_global_step()
        # 3.请确保训练数据是每个worker都能够访问得到的
        X,y = dataset_iterator.get_next()
        # inference step
        y_logits = inference(X)
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits,
                                                             labels=y)
        loss = tf.reduce_mean(entropy)
        train_op = optimizer.minimize(loss,global_step = global_step)
        with tf.name_scope('init_and_save'):
            init = tf.global_variables_initializer()
            merged = tf.summary.merge_all()
        ''' 4. 使用 hook 管理 1: 总训练步数
                            2: 保存的文件路径和保存checkpoint的频率
                            3: 保存summary 
        <--- saver = tf.train.Saver()'''
        hooks=[tf.train.StopAtStepHook(last_step=params['train_steps']),
               tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.checkpoint_dir,
                                            save_steps=50,
                                            saver=saver),
               tf.train.SummarySaverHook(save_steps=50,summary_op=merged,
                   output_dir=FLAGS.summaries_dir + str(FLAGS.model_version) + '/train')]
        ''' 5. 使用 tf.train.MonitoredTrainingSession 代替 tf.Session。需要填写主worker 的is_chief 参数，表示该节点的worker会负责参数的初始化，保存等工作
        <--- with tf.Session() as sess:'''
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               hooks = hooks,) as mon_sess:   
            # 6.一些适应的改变
            mon_sess.run(init)
            local_step = 0
            while not mon_sess.should_stop():
            	_,losses,g_step =mon_sess.run([train_op,loss,global_step])
                local_step += 1
                if local_step % 1000 ==0:
                	print ('Worker %d: traing step %d done (global step:%d), loss is %f' % (FLAGS.task_index,local_step,g_step,losses))
    

需要说明的是，(i) tf.train.MonitoredTrainingSession() 是一个类似于session 的包装类，并没有继承tf.Session，所以一些sess的操作不一定有，不过里面是有一个session的元素。

(ii) 保存checkpoint 也不一定使用hook 保存，可以参考网上的其他的一些资料进行自适应的改动，不过分布式训练需要注意的是，训练完成过后，不同节点会自行关闭该进程，不同节点训练模型参数怎么传回去给主worker我还不是很清楚，如果使用单机的保存方式，可能会有问题。

(iii) 这里我们没有给出训练使用的是同步更新还是异步更新权重，默认是异步更新的，如果又需要的话可以查看 tf.train.SyncReplicasOptimizer() 这个接口，看如何设置同步/异步

	以上是一些从单机训练转成分布式训练需要作的一些改动，不过直接使用以上的方法改动，在大的集群上，不一定有很高的加速比，这个需要更加深入的讨论，这里不涉及。另外，tensorflow 自带的estimator 接口也提供更加高级的分布式训练接口，有兴趣的可以去研究一下。

使用clever 分布式改动

	clever 默认是使用kubeflow上的 tfjob 创建tensorflow 训练服务的，不同的节点在创建的时候会得到一个相同的环境变量 'TF_CONFIG'，我们只需要在前面相应的位置进行修改，就可以在clever平台上跑起来了。

前情回顾：

    # 首先需要给不同的节点分配不同的工作
    tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
    tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
    FLAGS = tf.app.flags.FLAGS
    
    ## 这里ps-**指代的是不同节点的ip地址
    
    '''<--- parameter_servers = ["pc-01:2222"]
    workers = [	"pc-02:2222", 
    			"pc-03:2222",
    			"pc-04:2222"]
    原来的时候需要手动设集群的ip和port，使用tf_job后台会自动管理'''
     
      
    --------------------------------------># 改动的位置
    ## 通过环境变量设置 job_name, task_index
    tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
    task_config = tf_config.get('task', {})
    task_type = task_config.get('type')
    task_index = task_config.get('index')
    FLAGS.job_name = task_type
    FLAGS.task_index = task_index
    
    ### 确认不同的节点创建没有问题
    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print ('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print ('task_index : %d' % FLAGS.task_index)
        print("job name = %s" % FLAGS.job_name)
        print("task index = %d" % FLAGS.task_index)
    
    # 填写cluster,ps,woker等信息
    cluster_config = tf_config.get('cluster', {})
    ps_hosts = cluster_config.get('ps')
    worker_hosts = cluster_config.get('worker')
    
    ps_hosts_str = ','.join(ps_hosts)
    worker_hosts_str = ','.join(worker_hosts)
    
    FLAGS.ps_hosts = ps_hosts_str
    FLAGS.worker_hosts = worker_hosts_str
    
    parameter_servers = FLAGS.ps_hosts.split(',')
    workers = FLAGS.worker_hosts.split(',')
    <----------------------------------------- # 下面与之前一样
    
    # 这里相当于告诉tensorflow 这几台机子作为一个集群了(不同节点是同一份集群的名单)
    cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})
    ### 不同节点负责不同工作
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

其实主要的工作就是在每一个节点上读取环境变量'TF_CONFIG' 内容然后找到属于自己的config把服务起起来，可以省去很多繁琐的步骤。



参考

1. https://github.com/ischlag/distributed-tensorflow-example 这是一个更基础的分布式例子，不过是基于tf1.2的有一些接口可能已经弃用了。
2. https://github.com/uber/horovod uber 维护的一个分布式训练框架，除了tensorflow 还支持pytorch 等深度学习框架的分布式部署，kubeflow 中的 mpi_job 也是基于horovod 的。
3. https://github.com/tmulc18/Distributed-TensorFlow-Guide 这是一个更加细致的分布式使用例子，有很多的参数可以调，有需要可以自己尝试一下
