IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 24, NO. 9, SEPTEMBER 2023

# Yalan Wu, Jigang Wu, Member, IEEE, Mianyang Yao, Bosheng Liu, Long Chen, and Siew Kei Lam, Senior Member, IEEE

# Abstract

In vehicular networks, task scheduling at the microarchitecture-level and network-level offers tremendous potential to improve the quality of computing services for deep neural network (DNN) inference. However, existing task scheduling works only focus on either one of the two levels, which results in inefficient utilization of computing resources. This paper aims to fill this gap by formulating a two-level scheduling problem for DNN inference tasks in a vehicular network, with an objective of minimizing total weighted sum of response time and energy consumption for all tasks under the following constraints: per task response time, per vehicle energy consumption, per vehicle storage capacity. We first formulate the problem and prove that it is NP-hard. A group transformation based algorithm, called GTA, is proposed. GTA makes scheduling decisions at the network-level using the group transformation based approach, and at the microarchitecture-level using a greedy strategy. In addition, an algorithm, denoted as DRL, is proposed to decrease total weighted sum of response time and energy consumption for all tasks. DRL trains two models with deep reinforcement learning to achieve two-level scheduling. The proposed algorithms are evaluated on a platform consisting of a desktop, Raspberry Pi, Eyeriss, OSM, SUMO, NS-3. Simulation results show that DRL outperforms the state-of-the-art methods for all cases, while the proposed GTA outperforms the state-of-the-art methods for most cases, in terms of total weighted sum of response time and energy consumption. Compared with four baseline algorithms, GTA and DRL reduce the total weighted sum of response time and energy consumption by 41.49% and 62.38%, on average respectively, for different numbers of tasks.

# Index Terms

Vehicular network, two-level task scheduling, DNN inference, quality of computing services, accelerator.

# I. INTRODUCTION

RECENT years have seen the emergence and advancement of many driving assistance applications, such as pedestrian detection, obstacles avoidance, augmented reality navigation etc. [1], [2]. These applications now commonly rely on deep learning techniques (e.g., deep neural networks (DNNs)) to achieve higher performance. In practical systems, the computing services for these applications perform inference using pre-trained DNN models. The response time and energy consumption are two critical factors for evaluating the quality of computing services (QoS) [3], [4], [5]. Response time is not only important to ensure safety and improve driving experience, but it can also impact traffic congestion. In the European Union, road transportation is the second largest source for CO2 emissions in 2018 [6], and it could become the largest source of CO2 emissions in 2050 [7]. Moreover, overheating caused by high energy consumption may degrade the effectiveness and lifespan of the computing units on vehicles [8], [9]. As such, it is imperative to provide computing services with low response time and energy consumption for DNN inference tasks in vehicular networks. However, DNN inference algorithms are computationally intensive, which require massive resources for computation, communication, and storage, etc. [10]. These DNN inference tasks generally suffer from poor QoS on vehicles due to the limited in-vehicle computing resources. Task scheduling at the microarchitecture-level and network-level has been separately explored to improve QoS of DNN inference in vehicular networks [10], [11].

As shown in Fig. 1(a), task scheduling at microarchitecture-level exploits the heterogeneous computing capability in a vehicle, which may consist of central processing unit (CPU), graphics processing unit (GPU), digital signal processing, or DNN accelerators (denoted as accelerators for simplicity) [12], [13]. QoS for a DNN inference task can be improved by allocating parts of the task to different computing units so that they can be executed in parallel [14], [15]. From Fig. 1(b), task scheduling at the network-level promotes QoS by allocating part of the DNN inference tasks to other vehicles [16], [17]. Specifically, we can schedule inference tasks from one vehicle to another more powerful vehicle to achieve acceleration or energy saving [18], [19]. Meanwhile, the tasks on different vehicles can be processed in parallel to further improve QoS. Especially, it has more advantages to schedule tasks among vehicles than to schedule tasks between vehicle.

Digital Object Identifier 10.\frac{1109}{TITS}.2023.3266795

Manuscript received 14 January 2022; revised 20 October 2022 and 11 January 2023; accepted 10 April 2023. Date of publication 19 April 2023; date of current version 30 August 2023. This work was supported in part by the National Natural Science Foundation of China under Grant 62202108 and Grant 62072118; in part by the Guangdong Basic and Applied Basic Research Foundation under Grant 2021B1515120010; in part by the Guangdong Natural Science Foundation under Grant 2023A1515030183, Grant 2022A1515010895, and Grant 2023A1515011230; and in part by the Huanpu International Science and Technology Cooperation Foundation under Grant 2021GH12. The work of Yalan Wu was supported by Nanyang Technological University through the Post-Doctoral Project for International Training of Guangdong. The Associate Editor for this article was Z. Cao. (Corresponding author: Jigang Wu.)

Yalan Wu is with the School of Integrated Circuits, Guangdong University of Technology, Guangzhou 510006, China, and also with the School of Computer Science and Engineering, Nanyang Technological University, Singapore 639798 (e-mail: asyalanwu@outlook.com).

Siew Kei Lam is with the School of Computer Science and Engineering, Nanyang Technological University, Singapore 639798.

1558-0016 © 2023 IEEE. Personal use is permitted, but \frac{republication}{redistribution} requires IEEE permission. See https://www.ieee.\frac{org}{publications}/\frac{rights}{index}.html for more information.

WU et al.: TWO-LEVEL SCHEDULING ALGORITHMS FOR DNN INFERENCE IN VEHICULAR NETWORKS

# II. MOTIVATIONS AND CHALLENGES

Existing works on inference task scheduling at microarchitecture-level limit the potential improvement in QoS, as the computing resource in a single vehicle is limited. On the other hand, in existing works at network-level, the in-vehicle computing resources cannot be perfectly utilized. To the best of our knowledge, there is no report in QoS, as the computing resource in a single vehicle is limited.

We provide a motivational example by simulating task scheduling at different levels. Fig. 2 shows the scheduling results on different types of DNN inference tasks, in terms of total weighted sum of response time and energy consumption for all tasks (which is defined as total trade-off cost in this paper, as shown in definition 1 of section V). The corresponding DNN models for inference tasks are MobileNetV3-Small, ResNet18 and VGG16. The parameter setting for the simulations is shown in Fig. 2(a). Note that all scheduling strategies at different levels are based on brute-force approach. For simplicity, in Fig. 2(b), the scheduling strategies at microarchitecture-level, network-level and two-level, are denoted as Microarchitecture, Network and Two-level, respectively.

It can be seen from Fig. 2(b), the total trade-off cost for Two-level significantly decreases for different types of DNN inference tasks, compared with Microarchitecture or Network. Specifically, for the VGG16 model, the total trade-off cost for Two-level is decreased by 62.78% and 43.76%, compared with Microarchitecture and Network, respectively.

There are two major challenges for designing two-level scheduling algorithms in vehicular networks.

# Challenge I:

The scheduling decisions at different levels are inter-dependent. Specifically, at microarchitecture-level, different scheduling decisions lead to different qualities of computing services on a vehicle. Additionally, the qualities can influence the scheduling decisions at network-level. Similarly, at network-level, different scheduling decisions lead to different parts of inference tasks executed on a vehicle. Note that an accelerator is customized for a special application. For example, the accelerator Eyeriss [21] is customized for deep convolution neural networks, and it only can accelerate the convolution layer or fully connected layer. As a result, different parts of inference tasks may dominate the quality improvement for computing services on a vehicle.

# Challenge II:

High mobility of vehicles makes it difficult to improve QoS for DNN inference tasks using two-level scheduling algorithms. The high mobility of vehicles leads to high dynamic topology and cooperation relationship among vehicles, which has great impact on the transmission rate and stability of communication connection in a vehicular network. Specifically, the fast-moving vehicles in vehicular network give rise to rapid changes in relative locations between two vehicles, which in turn results in rapid changes in transmission rate between two vehicles. Moreover, the high mobility of vehicles may result in frequent handover and unstable communication connection between two vehicles. As a result, the offloading decisions based on current state may not be suitable for the next state in vehicular networks.

To address challenge I, we jointly schedule DNN inference tasks at the microarchitecture and network levels for improving QoS. Firstly, we make scheduling decisions for DNN inference tasks from network-level to microarchitecture-level. We use group family to represent the current state.

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18, 2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 24, NO. 9, SEPTEMBER 2023

in a vehicular network. Then, we traverse all task groups for each task to schedule the tasks at network-level, where the scheduling decisions at microarchitecture-level are made based on greedy strategy in the traversal process. Secondly, we make scheduling decisions for DNN inference tasks from microarchitecture-level to network-level. Specifically, we generate scheduling decisions at microarchitecture-level on each vehicle in advance for each DNN inference task based on deep reinforcement learning. After that, we produce scheduling decisions for network-level based on the decisions generated at the microarchitecture-level.

To address challenge II, we will explore an efficient algorithm based on current state to alleviate side effects caused by high mobility of vehicles. Specifically, each task is indivisible at the network-level. At the microarchitecture-level, we schedule layers of tasks based on a greedy strategy to make rapid decisions for the two-level scheduling. It is well known that deep reinforcement learning enables cognitive capability in dynamic environment and decision making [22]. Thus, we develop an algorithm that exploits deep reinforcement learning to make scheduling decisions at two-level, in order to accommodate to highly dynamic environment. To accelerate the algorithm, we also design a binding operation to reduce the action space for task scheduling at network-level.

# The main contributions of this paper are as follows.

- Novel formulated problem. A two-level scheduling problem is formulated for DNN inference tasks in a vehicular network. The objective is to minimize the total weighted sum of response time and energy consumption for all tasks, under the constraints of response time, energy consumption, and storage capacity, etc. In addition, we also prove that the formulated problem is NP-hard.
- Group transformation based algorithm. A group transformation based algorithm is proposed for solving the formulated problem, by scheduling DNN inference tasks at network-level based on group transformation rule, and by scheduling each layer of each task at microarchitecture-level based on greedy strategy.
- Deep reinforcement learning based algorithm. A deep reinforcement learning based algorithm is proposed for the formulated problem. Initially, a model is pre-trained to make scheduling decisions at microarchitecture-level for each layer on each vehicle based on deep reinforcement learning. Then, binding operation is performed for each DNN model to reduce the exploration space. Finally, two-level scheduling decisions are generated, by employing deep reinforcement learning.
- Efficiency of the proposed algorithms. We integrated a desktop, Raspberry Pi, Eyeriss, OSM, SUMO and NS-3 to evaluate the performance of algorithms. Simulation results show that the proposed algorithms outperform the state-of-the-art methods for most cases, both in total weighted sum of response time and energy consumption for all tasks and in the proportion of failed tasks. In particular, the proposed deep reinforcement learning based algorithm performs better than the state-of-the-art methods for all cases, in terms of total weighted sum of response time and energy consumption.

The rest of this paper is organized as follows. Section III discusses the related works. Section IV describes the system model. Section V formulates the proposed problem. Section VI presents scheduling algorithms. Section VII shows simulation results and performance analysis. Finally, Section VIII concludes the paper.

# III. RELATED WORKS

In vehicular networks, task scheduling is a promising technique to enhance QoS for the DNN inference tasks, where scheduling at microarchitecture-level and at network-level for the tasks are separately explored.

# A. Scheduling at Microarchitecture-Level

For task scheduling at microarchitecture-level, existing works mainly focus on improving the performances in terms of response time and energy consumption. For minimizing response time of DNN inference tasks, a real-time framework for task scheduling is proposed by facilitating co-utilization of CPU and GPU to accelerate DNN inference in a multi-core system [23]. A task scheduling strategy is proposed to accelerate graph convolutional network (GCN) inference by scheduling task to digital signal processing based on hardware pipeline in their designed field-programmable gate array (FPGA) architecture [24]. For improving energy-efficiency, an input-aware framework is proposed by scheduling individual training operations to GPU or FPGA accelerator on a heterogeneous system with GPUs and FPGAs [25]. A full-stack \frac{hardware}{software} heterogeneous system is designed, to achieve high throughput and low energy consumption by scheduling task to CPU or FPGA for scene text detection [26]. In addition, an in-situ autonomous and incremental computing framework is proposed, to save energy by scheduling inference and diagnosis tasks to mobile GPU and FPGA [15].

On other aspects, there are some interesting works. For example, the online resource management approaches are proposed to trade off various performance metrics by scheduling tasks to different computing units on heterogeneous multi-core system [27]. A quality of experience aware scheme for task scheduling is proposed by dynamically reconfiguring FPGA and selecting DNN model for different requirements [28].

# B. Scheduling at Network-Level

For task scheduling at network-level, most existing works focus on performance improvement on response time [29], [30] and energy consumption [31], [32]. For minimizing total response time of DNN inference tasks, Mohammed et al. propose a distributed algorithm based on matching theory by jointing partitioning and scheduling for DNN inference tasks in fog networks [17]. Wang et al. propose a framework for vehicle and edge collaborative computing by partitioning DNN inference tasks and scheduling part of tasks, to minimize the overall latency of all vehicles [33]. He et al. propose an algorithm by deploying DNN partition model and scheduling.

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18,2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

WU et al.: TWO-LEVEL SCHEDULING ALGORITHMS FOR DNN INFERENCE IN VEHICULAR NETWORKS

# TABLE I

# MAIN NOTATIONS IN SYSTEM MODEL

Fig. 3. The two-level architectures for scheduling on DNN inference tasks in a vehicular network.

DNN inference task, to reduce end-to-end inference delay [34]. Wu et al. propose a deep reinforcement learning algorithm for sampling rate adaption, inference task scheduling and resource allocation, to minimize the average service delay with guaranteeing long-term accuracy requirement [35]. Wu et al. propose a network model for directional vehicle mobility and efficient algorithms for task scheduling to decrease total delay of tasks [20]. For minimizing energy consumption, Xu et al. propose an approximation algorithm, followed by a learning based algorithm, for DNN inference task scheduling among base stations and cloudlets [31]. In addition, Li et al. propose a decomposition-oriented approach for task scheduling in multi-user edge networks by exploiting alternating direction method of multipliers, to decrease the weighted summation of energy consumption and execution latency [32].

On other aspects, there are some interesting works. For example, Tan and Cao propose heuristics based solutions for scheduling deep learning video analytics task to NPU in mobile device or edge server, to maximize the system utility or accuracy [36]. Shlezinger et al. propose a decentralized mechanism for collaborating DNN inference with edge devices, to improve the accuracy [37]. Huang and Zhou propose a dynamic scheme for compression ratio selection and a scheme for information augmentation to maximize the expectation of the number of completed tasks [38]. Wu et al. propose a series of algorithms, such as approximation algorithm, deep reinforcement learning based algorithms and coalition based algorithms, for achieving load balance by scheduling tasks to neighbor vehicles [5].

# A. Vehicle Details

Generally, vehicles are heterogeneous in terms of the equipped computing units, storage capacity, and energy capacity, which is caused by the different vehicle types existing in a vehicular network. For example, some vehicles are only equipped with a CPU, while some vehicles are equipped with a CPU and a DNN accelerator (denoted as accelerator in the following for simplicity). Assume that accelerator can only work on convolution layers and fully connected layers of DNN inference tasks in this paper, according to the characterize of accelerator Eyeriss [21]. It is noteworthy that a vehicle may be equipped with several computing units, such as CPU, GPU, and accelerator, etc. [12], [13]. Specifically, CPU and GPU are universal computing units, while accelerator is special computing unit. Therefore, for simplicity, this paper considers that a vehicle is equipped with a CPU and an accelerator which is customized for deep convolution neural network at most.

# IV. SYSTEM MODEL

We consider a vehicular network with n vehicles in a road segment, as shown in Fig. 3. It is worth mentioning that, task scheduling to provide high QoS for DNN inference tasks.

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18,2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 24, NO. 9, SEPTEMBER 2023

For example, as shown in Fig. 3, vehicle v7 is only equipped with a CPU, while vehicle v8 is equipped with a CPU and an accelerator. The system model can easily fit for more equipped computing units in a vehicle.

Let N = {1, 2, \cdot \cdot \cdot , n}, and vk be a vehicle in the road segment for k \in N. Then, let V be the set of vehicles in a road segment, where V = {v1, v2, \cdot \cdot \cdot , vn}. For each vehicle vk, we use a tuple {Ekmax(t), qkmax(t), uk, fkcpu, fkacc} to describe the details, where k \in N. Here, Emax(t) and qkmax(t) represent the maximum tolerant energy consumption and available storage capacity, respectively, for vehicle vk at time t. Meanwhile, uk (uk \in {0, 1}) indicates whether vehicle vk is equipped with an accelerator, where uk = 1 (or uk = 0) represents that vk is (or is not) equipped with an accelerator. For simplify, the accelerators on different vehicles are assumed to be homogeneous in this paper. Besides, fkcpu and fkacc are the clock frequency for CPU and accelerator, respectively, in vk. Note that, fkacc is set to null value if uk = 0. In addition, there is a task queue buffer in each vehicle to cache tasks waiting to be executed. Tasks in task queue buffer are orderly processed on the manner of first in first out (FIFO).

At time t. Thus, xi jk(t) can be defined as,

xi jk(t) =
1, if ri j(t) is scheduled to vk,

We use a set X(t) to record the scheduling decisions at network-level for all DNN inference tasks at time t, i.e.,

X(t) = {xi jk(t) | 1 \leq i \leq m, 1 \leq j \leq li, 1 \leq k \leq n}.

In addition, the parts of a DNN inference task can be scheduled to CPU or accelerator on a vehicle at microarchitecture-level, if the vehicle owns an accelerator. Let yi jk(t) denote the binary scheduling indicator at microarchitecture-level between layer ri j(t) and vehicle vk at time t. Thus, yi jk(t) can be defined as,

yi jk(t) =
1, if ri j(t) is scheduled to the accelerator in vk,

Note that yi jk(t) = 0 if uk = 0, for i \in M and j \in Li. We use a set Y(t) to record the scheduling decisions at microarchitecture-level for all DNN inference tasks at time t, i.e., Y(t) = {yi jk(t) | 1 \leq i \leq m, 1 \leq j \leq li, 1 \leq k \leq n}.

# B. Task Details

In a vehicular network, some vehicles generate DNN inference tasks during a certain time period T for navigating, pedestrian detection, entertainment, etc. We assume that all inference tasks are not sensitive to data privacy. Let M = {1, 2, \cdot \cdot \cdot , m}. These generated DNN inference tasks are denoted as R(t) = {r1(t), r2(t), \cdot \cdot \cdot , rm(t)}, where ri(t) indicates a DNN inference task for i \in M, and m is the number of tasks at time t (t \in T). For each task ri(t), we use (t) and Timax(t) to denote the vehicle which generates the task and maximum tolerant response time of the task, respectively.

A DNN inference task has a layer-wise structure, which can be regarded as a directed acyclic graph with multiple layers. Each layer can be regarded as a subtask for a DNN inference task. Let Li = {1, 2, \cdot \cdot \cdot , li} and li is the number of layers for task ri(t). Thus, each DNN inference task ri(t) can be represented by ri(t) = ⟨ri1(t), ri2(t), \cdot \cdot \cdot , ric(t)⟩.

# C. Computation Delay and Communication Delay

In this paper, each layer of a DNN inference task can be processed on a CPU or an accelerator at microarchitecture-level. Meanwhile, it also can be processed on the local vehicle or other vehicles at network-level. Thus, let Ti cj(t) be the computation delay for processing layer ri j(t) of DNN inference task ri(t), which is defined by,

T c i j(t) =
wcpu i j(t)/ fkcpu, xi jk(t) = 1 &#x26; yi jk(t) = 0,
wi accj(t)/ fkacc, xi jk(t) = 1 &#x26; yi jk(t) = 1.

For network-level, the wireless channels between two vehicles are assumed to flat fading. Let s be the small-scale fading channel power gain from zk indicate to vk, which obeys exponentially distributed vehicle vz i.e., szk ∼ Exp(1). Thus, the channel power gain between vz and vk, denoted as Gzk(t), is defined as follows.

Gzk(t) = szk \cdot g^{\prime} \cdot (d^{\prime})\theta.

For each layer of DNN inference, we use a tuple {di inj(t), di outj(t), wi cpuj(t), wi accj(t)} to denote the details. Here, di inj(t) and di outj(t) are the input data size and output data size, respectively, for ri j(t). Besides, wi cpuj(t) and wi accj(t) are the workload of ri j(t) on CPU and accelerator, respectively. It is worthwhile to note that, we assume that the DNN models are pre-loaded on all vehicles in this paper. Meanwhile, wi accj(t) = +\infty if the layer is not convolution layer or fully connected layer. This is because the accelerators are customized for the convolution layers or fully connected layers.

In this paper, the parts of a DNN inference task derived from vehicles can be scheduled to other vehicles or be processed locally at network-level. Let xi jk(t) be the binary scheduling indicator at network-level between layer ri j(t) and vehicle vk where g^{\prime}, d^{\prime}, \theta and dzk(t) are the path-loss constant, reference distance, path-loss exponent and the distance between vz and vk at time t, respectively.

According to the channel power gain, the transmission rate between vz and vk, which is denoted as hzk(t), is defined by,

hzk(t) = Bz \cdot ln(1 + p \cdot Gzk(t) / N0).

Here, Bz is the channel bandwidth owned by vehicle vz, p is the transmit power of vehicle, and N0 is the power of Additive White Gaussian Noise (AWGN). In this paper, vehicles are homogeneous in terms of transmit power.

Note that, at network-level, the communication delay for a DNN inference task is comprised of two parts: one is for transmitting the input data of each layer, and the other is\dots

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18, 2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

WU et al.: TWO-LEVEL SCHEDULING ALGORITHMS FOR DNN INFERENCE IN VEHICULAR NETWORKS

for transmitting the output data of last layer to the vehicle generated the task. Meanwhile, at microarchitecture-level, the communication delay for a DNN inference the communication overhead between CPU and accelerator for transmitting input data of each layer and output data of the last layer, if the corresponding layer is scheduled to accelerator.

For convenience, we use \eta_{ik}(t) to indicate whether vk vehicle which generates the DNN inference task ri(t), which is defined by,

\eta_{ik}(t) =

{

1, if vehicle vk does not generate ri(t),

}

For the first layer, the input data is required to transmit to vehicle vk, if the first layer is scheduled to vk at network-level. Let Tiin1(t) be the communication delay for transmitting the input data of the first layer at network-level. Then, Tiin1(t) is defined by,

Tiin1(t) = \sum_{k=1}n xi1k(t) \cdot \eta_{ik}(t) \cdot din1(t),

where \tau_{i} inj(t) is defined by,

\tau_{i} inj(t) =

{

\tau_{iin}1(t), j = 1,

\sum_{k=1}n yi jk(t) \cdot \zeta_{i} j \cdot i bjk, 1 &#x3C; j \leq li,

}

It is noteworthy that, the output of the last layer for task ri(t) is feedback, which is required to return to the vehicle which generates ri(t). Thus, the communication delay for ri(t) consists of two parts: one is for transmitting the feedback to the vehicle which generates the task, and the other is for transmitting the intermediate parameters. For the last layer, we use Ticout(t) to represent the communication delay for transmitting the feedback from vehicle vk to vehicle oi(t) generated ri(t) at network-level, where c = li. Then, it is defined by,

Ticout(t) = \sum_{k=1}n xick(t) \cdot \eta_{ik}(t) \cdot dout(t),

where t^{\prime} = ts(t) + Tw(t) + Tc(t).

Let \tau_{iin}1(t) be the communication delay for transmitting the input data of the first layer at microarchitecture-level. Then, it is defined by,

\tau_{iin}1(t) = \sum_{k=1}n yi1k(t) \cdot din1(t),

where bk is the read bandwidth for DRAM of vk. Let Twj(t) be the waiting time processed in task queue buffer. Thus, the start time executing the first layer can be calculated by,

Tinet(t) = Tiin1(t) + Ticout(t),

where c = li. Meanwhile, \tau_{imi}c(t) is defined by,

\tau_{mic}(t) = \sum_{j=1}li \tau_{in}(t) + \tau_{out}(t),

where c = li j=1. Let Tim(t) be the communication delay for ri(t), including the delay both at network-level and at microarchitecture-level. According to (15) and (16), Tim(t) is defined by,

Tim(t) = Tinet(t) + \tau_{imi}c(t).

Let Ti(t) be the response time of the DNN inference task ri(t). The response time of a task is defined as the time period from the time when the task arrived to the time when the task completed. Thus, response time of ri(t) is comprised of communication delay and waiting time for Ti(t) is defined as,

Ti(t) = \sum_{j=1}li Tcj(t) + Tim(t) + Tw(t),

where t^{\prime} = tisj^{\prime}(t) + Twj^{\prime}(t) + Tcj^{\prime}(t) and j^{\prime} = j - 1. in task queue i is the total waiting time for ri(t) to be processed in task queue buffer.

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18, 2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 24, NO. 9, SEPTEMBER 2023

# D. Energy Consumption

Let  Eicj(t) be energy consumption for executing the layer rij(t) of DNN inference task ri(t). If layer rij(t) is executed on CPU, Eicj(t) is dominated by CPU clock frequency of vehicle vk which executes rij(t). Otherwise, Eicj(t) is dominated by the processing speed for rij(t) on an accelerator. Thus, Eicj(t) is defined by:

Ec(t) = \mu \cdot wicpuj(t) \cdot (fkcpu)2, xijk(t) = 1 &#x26; yijk(t) = 0,

wiaccj(t) \cdot ek, xijk(t) = 1 &#x26; yijk(t) = 1. (19)

Here, \mu is effective capacitance, which depends on the chip architecture, it is set to 10-27 by default [41]. ek is energy consumption per cycle for accelerator on vehicle vk, which depends on the designed architecture of an accelerator.

For network-level, let Einet(t) be the energy consumption for scheduling ri(t). According to [42], Einet(t) is dominated by the transmit power of vehicle and the communication delay at network-level. Thus, Einet(t) is defined by:

\epsilon_{km2}(t) = p \sum_{i=1} m \sum_{j=1} l xijk(t) \cdot (1 - xizk(t)) \cdot dhioutj(t(t^{\prime})) + xick(t) \cdot ka Ticout(t), (26)

where z = j + 1, va is the vehicle executed the next layer, and t^{\prime} = ts + Tc(t).

For network-level, let \epsilon_{net}(t) be the energy consumption for vehicle vk to transmit the input or output data of DNNk inference tasks. \epsilon_{kne}t(t) includes the energy consumption for three parts, which are for transmitting input or output data of the first layer, middle layer, and the last layer, respectively.

Einet(t) = p \cdot Tinet(t). (20)

For microarchitecture-level, let Eimic(t) be the energy consumption for scheduling ri(t). Eimic(t) is dominated by the architecture of accelerator and the communication delay at microarchitecture-level. Thus, Emic(t) = e^{\prime} \cdot i \tau(t) can be defined by:

Emic(t) = e^{\prime} \cdot iimic(t). (21)

where e^{\prime} is energy consumption per second for transmitting data from CPU to the accelerator on a vehicle.

Let Eim(t) be the energy consumption for scheduling ri(t), which includes energy consumption both at network-level and at microarchitecture-level. Thus, Eim(t) can be calculated by:

Eim(t) = Einet(t) + Eimic(t). (22)

Therefore, the energy consumption for completing DNN inference task ri(t), denoted as Ei(t), is defined by:

Ei(t) = \sum_{j=1}l Eicj(t) + Eim(t). (23)

It is well known that, there exists the maximum tolerable energy consumption for each vehicle, due to the limited energy capacity and preventing overheat of computing units [8], [9]. Thus, we analyze the energy consumption for each vehicle as follows.

Let \epsilon_{kc}(t) be the energy consumption for vehicle vk to execute DNN inference tasks. Then, it is defined by:

\epsilon_{kc}(t) = \sum_{i=1} m \sum_{j=1} l xijk(t) \cdot Eicj(t). (24)

Let \epsilon_{kmi}c(t) be the energy consumption on vk for transmitting data at microarchitecture-level, which is calculated by:

\epsilon_{kmi}c(t) = \epsilon_{kin}(t) + \epsilon_{kou}t(t). (30)

Let \epsilon_{k}(t) be the energy consumption on vehicle vk for completing the DNN inference tasks in vehicular network at time t. Then, \epsilon_{k}(t) can be calculated by:

\epsilon_{k}(t) = \epsilon_{kc}(t) + \epsilon_{kne}t(t) + \epsilon_{kmi}c(t). (31)

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18, 2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

WU et al.: TWO-LEVEL SCHEDULING ALGORITHMS FOR DNN INFERENCE IN VEHICULAR NETWORKS

# E. Storage Capacity

Definition 1: The total trade-off cost refers to the total weighted sum of response time and energy consumption for all inference tasks in a vehicular network. The problem, denoted as P in this paper, is formulated as follows.

min
\sum_{i=1}m (\alpha_{i} \cdot Ti(t) + \beta_{i} \cdot Ei(t)) (37)

s.t. Ti(t) \leq Timax(t), \foralli \in M, (38)

q\epsilon_{kk}(t) \leq Ekmax(t), \forallk \in N, (39)

qin(t) = yik(t) \cdot \sum_{l=1}i di1(t) + \sum_{j=2}l (yij(t) \cdot din(t)) (40)

where j^{\prime} = j - 1. Meanwhile, ςin(t) can be calculated by,

ςin(t) = (1 - yi1k(t)) \cdot \eta_{ik}(t) \cdot xiik1k(t) \cdot din(t) + \sum_{j=2}l (1 - yij(t)) \cdot xij(t) \cdot \zeta_{ij} \cdot din(t) (33)

Let qout(t) and ςout(t) be the occupied storage capacity on vehicle for the output data of ri(t) at time t at microarchitecture-level and at network-level, respectively. qout(t) is defined by,

qout(t) = \sum_{j=1}l yij(t) \cdot (1 - yiz(t)) \cdot dout(t) + yick(t) \cdot dout(t) (34)

where z = j + 1 and c = li. Similarly, ςout(t) is defined by,

ςout(t) = \sum_{j=1}l (1 - yij(t)) \cdot xij(t) \cdot \zeta_{iz} \cdot dout(t) + (1 - yick(t)) \cdot \eta_{ik}(t) \cdot xick(t) \cdot dout(t) (35)

Let qk(t) be the required storage capacity on vehicle for processing DNN inference tasks at time t. According to (32)-(35), qk(t) can be calculated by,

qk(t) = \sum_{i=1}m (qin(t) + ςin(t) + qout(t) + ςout(t)) (36)

In other words, the later layer rij^{\prime}(t) must be executed after the preceding layer rij(t) is completed and the required input data is transferred to the corresponding place executed rij^{\prime}(t).

# V. PROBLEM FORMULATION

Generally, time and energy costs are critical impact factors for QoS of a DNN inference task. Besides, the costs of time and energy are proportionate to the decreased QoS. In other words, the higher time and energy costs are, the worser QoS for a DNN inference task is. Moreover, each vehicle has its preference for response time of tasks or energy consumption of vehicles. Thus, this paper introduces the weighted sum of response time and energy consumption for DNN inference tasks to reflect QoS and the preferences of vehicles.

In order to improve QoS, we aim to minimize the weighted sum of response time and energy consumption for all DNN inference tasks. For simplicity, we introduce the total trade-off cost, which is defined as follows.

max
\sum_{z=1}\kappa xz \cdot pz (46)

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18,2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 24, NO. 9, SEPTEMBER 2023

s.t.                                                                                                                                                    ```html
\sum\kappa xz \cdot \omega_{z} \leq C, (47)

xz \in {0, 1},         \forallz = {1, 2, \cdot \cdot \cdot , \kappa}. (48)

Let us consider a special case of problem P, in which the scheduling at microarchitecture-level is not considered. Moreover, the constraints (38), (39), (41), (43) and (45) are deleted. In the special case, each layer of DNN inference tasks is regarded as an item, the negative value of trade-off cost corresponds to the profit, and the occupied storage capacity is viewed as the weight. Meanwhile, the available storage capacity of a vehicle corresponds to the limitation of total weight of the knapsack. Therefore, the special case of problem P can be mapped into the 0-1 KP. It is well known that 0-1 KP is proved to be NP-complete [43], then the special case of problem P is also NP-complete. Thus, we conclude that the problem P is NP-complete.

It is easy to understand that the proposed problem P is a stochastic optimization problem, and it is a great challenge to handle a large amount of stochastic information in realistic traffic systems. In the system, the locations of vehicles are rapidly changed, and the number of vehicles which simultaneously transmit data is also changed across time. Thus, transmission rate between each two vehicles is varied across time. Besides, it is also very difficult to predict the stochastic information, due to high mobility of vehicles. Therefore, it is a challenge to generate optimal solutions of P without any pre-knowledge.

# VI. ALGORITHMS

To the best of our knowledge, existing methods for task scheduling focus on either one of microarchitecture-level or network-level. Thus, the existing methods are not able to be well applied for two-level inference tasks scheduling, due to the existing challenges shown in section II. Therefore, in order to deal with the challenges, we propose two algorithms by jointly scheduling inference tasks at microarchitecture-level and network-level. To solve problem P, we initially propose a group transformation based algorithm, denoted as GTA, to schedule DNN inference tasks at two-level based on the current system status. Then, we propose a deep reinforcement learning based algorithm, denoted as DRL, to generate two-level scheduling decisions for DNN inference tasks, in order to cope with the network dynamics caused by high mobility of vehicles.

# A. Group Transformation Based Algorithm

In this paper, we construct task groups for vehicles and group family to implement cooperatively completing inference tasks in a vehicular network. The task group and group family are defined as follows.

Definition 2: The set of the tasks located in a vehicle is called as a task group for the vehicle.

Definition 3: The group family is the set of all task groups in a vehicular network, where the task groups in the family must hold the following conditions.

- 1) One task belongs to only one task group;

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18, 2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

- 2) Otherwise, h¯ik is trade-off cost including computation cost and communication cost, where scheduling decisions at microarchitecture-level is also generated by GS.

WU et al.: TWO-LEVEL SCHEDULING ALGORITHMS FOR DNN INFERENCE IN VEHICULAR NETWORKS

Input: ri (t), vk ;

Output: Yik (t), \chi_{ik} ;

1. Initialize yi jk (t) := 0 for 1 \leq j \leq li ;
2. for j := 1 to li do
3. temp1 := \alpha_{i} \cdot(wi cpuj (t)/ fkcpu + Ti wjk (t)) + \beta_{i} \cdot\mu\cdotwi cpuj (t);
4. if ri j (t) is a convolution layer or a fully connected layer and
5. G^{\prime} = {G1, \cdot \cdot \cdot , Gk^{\prime} , \cdot \cdot \cdot , Gz^{\prime} , \cdot \cdot \cdot , Gn}.
6. In the two group families, task ri (t) is located in the task group Gk, Gk^{\prime} = Gk - {ri (t)}, and Gz^{\prime} = Gz \cup {ri (t)}.
7. For convenience, we give the following symbol definition.
8. Definition 5: G r\Rightarrowⁱ(t) G^{\prime} represents that the group family G^{\prime} is preferred for task ri (t) compared with G, i.e., ri (t) tends to

if yi( j-1)k = 1 then
9. and if vjk is equipped with CPU and accelerator then join in Gz instead of G^{\prime}.
10. temp2 := \alpha_{i} \cdot (wacc(t)/ f acc + T w (t)) + \beta \cdot e \cdot wi j (t);
12. temp2 := \alpha_{i} \cdot(wi accj (t)/ fkacc+di inj (t)/bk +Ti wjk (t))+ \beta_{i} \cdot (ek \cdot wi accj (t) + e^{\prime} \cdot di inj (t)/bk );
14. if j = li then
15. temp2 := temp2+\alpha_{i} \cdotdi outj (t)/bk^{\prime} +\beta_{i} \cdote^{\prime}\cdotdi outj (t)/bk^{\prime} ;
17. if temp1 > temp2 then
18. yi jk (t) := 1, \chi_{ik} := \chi_{ik} + temp2;
20. \chi_{ik} := \chi_{ik} + temp1;
23. \chi_{ik} := \chi_{ik} + temp1;
26. return Yik (t), \chi_{ik} .

# Procedure 1 Pro(Gk): Calculating the Trade-Off Cost for Gk

Input: Gk ;

Output: h¯ (Gk );

1. h¯ (Gk ) := 0;
2. for each ri (t) \in Gk do
3. Yik (t), \chi_{ik} := GS(ri (t), vk );
4. if oi (t) = vk then
5. h¯ ik := \chi_{ik} ;
7. v j := oi (t), h¯ ik := \chi_{ik} + \alpha_{i} \cdot hdⁱⁱⁿjk¹ (t(t)) + \beta_{i} \cdot p \cdot hdⁱⁱⁿjk¹ (t(t));
9. h¯ (Gk ) := h¯ (Gk ) + \chi_{ik} ;
11. return h¯ (Gk ).

Let h¯ (Gk ) be the trade-off cost for Gk, which is calculated by,

h¯ (Gk ) = \sum h¯ ik .

According to definition 4, we have,

ϕ(Gk ) = -¯h(Gk ).

For illustration purposes, we assume that there are two group families, i.e.,

G = {G1, \cdot \cdot \cdot , Gk , \cdot \cdot \cdot , Gz, \cdot \cdot \cdot , Gn}

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18,2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 24, NO. 9, SEPTEMBER 2023

# Fig. 4. The framework of algorithm DRL.

# Algorithm 1 GTA: Group Transformation Based Algorithm

where ri (t) \in Gk, G^{\prime} = {G1, \cdot \cdot \cdot , Gk^{\prime} , \cdot \cdot \cdot , Gz^{\prime} , \cdot \cdot \cdot , Gn},

Input: V;

Output: X (t), Y(t);

1:  Initialize group family Ginit := {G1, G2, \cdot \cdot \cdot , Gn} by forming the tasks in each vehicle as a task group, i.e.,
Gk := {ri (t) | ri (t) \in R(t) and oi (t) = vk}, for k \in N;
2:  Calculate the valuation for each task group based on procedure 1, G := Ginit;
4:     for i := 1 to m do
5:        Gk := the task group where ri (t) is located in;
6:        for z := 1 to n do
7:               if z = k then
8:                   G^{\prime} := G, Gk^{\prime} := Gk - {ri (t)}, Gz^{\prime} := Gz \cup {ri (t)};
9:                   h¯ (Gk^{\prime} ) := Pro(Gk^{\prime}), h¯ (Gz^{\prime} ) := Pro(Gz^{\prime});
10:                   ϕ(G^{\prime}) := ϕ(G) + h¯ (Gk) + h¯ (Gz) - h¯ (Gk^{\prime}) - h¯ (Gz^{\prime});
11:                   if ϕ(G) &#x3C; ϕ(G^{\prime}) then
12:                       G := G^{\prime};
13:                   end if
14:               end if
15:        end for
16:     end for
17:  until transformation operation terminates;
18:  for k := 1 to n do
19:     for each ri (t) \in Gk do
20:        yi jk (t) := 1 for 1 \leq j \leq li;
21:        Yik (t), \chi_{ik} := GS(ri (t), vk);
22:     end for
24:  Y(t) := \cupnk=1Yik (t);
25:  return X (t), Y(t).

# B. Deep Reinforcement Learning Based Algorithm

For scheduling process, the two-level scheduling decisions are made according to the current system state, such as computation ability, available storage capacity and real-time network status. Besides, the next system state only depends on the current state and scheduling decisions, instead of historical state and scheduling decisions. Thus, the dynamic process for task scheduling can be serialized into a markov decision process (MDP). In this section, we propose a deep reinforcement learning based algorithm, denoted as DRL, to deal with network dynamic. Note that the convergence of DRL would be affected by the high dynamic environment. Thus, DRL focuses on the current system state to make decisions for task scheduling. Fig. 4 shows the framework of DRL. Specifically, DRL contains three main stages as follows.

# Stage 1 (Pre-Training):

DRL initially trains a model, denoted as PRE, to make scheduling decisions at microarchitecture-level for each task on each vehicle.

Hence, the stability of the resulting group family must be considered. Here, we define the stability of a group family.

# Definition 7:

We say a group family (G = {G1, \cdot \cdot \cdot , Gk , \cdot \cdot \cdot , Gz, \cdot \cdot \cdot , Gn}) is stable, if and only if, \forallri (t) \in R(t), G^{\prime} r\Rightarrowi(t) G for all Gz \in G - {Gk} \cup {\emptyset},

where 1 \leq z \leq Z and Z is the number of time steps. For step, STATE. Let szm be the system state in PRE at z-th time.

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18, 2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

WU et al.: TWO-LEVEL SCHEDULING ALGORITHMS FOR DNN INFERENCE IN VEHICULAR NETWORKS

convenience, let L be the set of all fully connected layers or convolution layers for each task. Note that only the layers in L can be processed in accelerator. Thus, in PRE, we focus on the layers in L. From the system model, the trade-off cost for each layer covers abundant information, such as computation and energy capacities, etc., which can adequately reflect the environment. Thus, in PRE, system state szm is defined by trade-off cost for each layer in L. Let ℓi j be trade-off cost for ri j (t) in L. It is defined by,

ℓi j = \alpha_{i} \cdot (Ti cj (t) + \tau_{i} inj (t) + \tau_{i} outj (t)) (52)

Therefore, system state szm also can be described as,

szm = {ℓi j | \forallri j (t) \in L} (53)

\bullet ACTION. Let azm be the action at z-th time step and Am the set of all possible actions. In this stage, we aim to generate scheduling decisions for each task on each vehicle, such that trade-off cost for the task executed on each vehicle is minimized. Thus, azm is defined by,

azm = yi jk (t) (54)

where ri j (t) \in L and vk \in V. Meanwhile, Am = {yi jk (t) | \forallri j (t) \in L, vk \in V}.

\bullet REWARD. Let R(szm , azm ) be the reward which is generated after executing action azm based on the current state szm. In PRE, the objective is to minimize the total trade-off cost for all tasks. Each action is expected to be efficient, so reward is defined as the difference between two total trade-off costs, which are generated before executing azm and that after executing azm, respectively, for all layers of tasks in L. Specifically, reward R(szm ,\sum_{az}m ) is defined by,

L(\theta_{z}) = E[(Qtarget - Q(szm , azm ; \theta_{z}))²] (60)

where \theta_{z} indicates a set of parameter of value network (i.e., current Q-network) at z-th time step. Besides, stochastic gradient descent approach is exploited to update parameters for value network. The parameters of target network are updated to be the parameters of value network every \iota steps.

In addition, the iterations for time steps terminates, if one of the following conditions are satisfied.

1. There is no improvement in performance during the 0 consecutive time steps.
2. The number of time steps reaches |Am| + 1 where |Am| indicates the size of the set Am.

# Stage 2 (Binding)

It is well known that there are plenty of layers for a DNN model, including convolution layers, fully connected layers, and activation layers, etc. Meanwhile, the computation delay of other layers is much smaller than that on the convolution layers and fully connected layers. Therefore, for accelerating convergence of algorithms, the layer with low computation delay and the layer with high computation delay are considered to bind together to schedule at network-level. For convenience, we define a block as follows.

Definition 8: A set of layers is called as a block, if they are binded together to schedule at network-level.

In PRE, deep deterministic policy gradient (DDPG) is introduced [22]. In order to approximately evaluate accumulated reward by applying azm at szm, we define Q(szm , azm ) as the action-value function, i.e., Q-function, of a state-action pair (szm , azm ). The value of Q(szm , azm ) is updated by exploiting temporal difference (TD) method, i.e.,

Q(sm , am ) = Q(sm , am ) + \gamma \cdot (R(sm , am ) + \delta \cdot maxa\inAm Q(sm+1, a) - Q(sm , am )) (56)

Here, \gamma and \delta are learning rate and discount factor, respectively, where 0 &#x3C; \gamma \leq 1 and 0 \leq \delta \leq 1. Besides, \epsilon-greedy strategy [22] is exploited to select action policy in PRE.

As shown in Fig. 4, there is a memory pool in PRE, to store scheduling experiences. Let D be the set of scheduling experiences in memory pool. A scheduling experience, denoted as \psi_{zm}, is defined by,

\psi_{zm} = (szm , azm , R(szm , azm ), szm+1) (57)

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18,2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

# Stage 3 (Training)

In this stage, we also construct a MDP for solving problem P. First, we define the environment in DRL as a vehicular network, with the scheduling decisions generated by PRE and DNN inference tasks pre-processed by binding operation. Then, we define the critical elements in DRL as follows.

\bullet STATE. Let sz be the system state in DRL at z-th time step. From system model, the trade-off cost for each DNN inference task also covers abundant information, to reflect the entire environment. For convenience, let ℓi^{\prime} be the trade-off cost for task ri (t), i.e.,

ℓi^{\prime} = \alpha_{i} \cdot Ti (t) + \beta_{i} \cdot Ei (t) (61)

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 24, NO. 9, SEPTEMBER 2023

# Procedure 2 Pro_{RL} (Environment, State, Action, Reward):

# Algorithm 2 DRL: Deep Reinforcement Learning Based Framework of Reinforcement Learning

1:         Initialize memory pool D with size \nu, action-value function and target Q-function with random weights \theta ;
2:     for epi := 1 to 3 do
3:      Initialize state  s1 observed from environment;
4:      f lag := 0;
5:      for z := 1 to Z     do
6:       Select a random action az with probability \epsilon or az := arg maxa\inA Q(sz, a; \theta_{z}) with probability 1 - \epsilon;
7:       Execute action az in emulator, observe reward R(sz, az) and new state sz+1 from environment;
8:       if R(sz, az) = -\infty then
9:            Break;
10:       end if
11:       Store (sz, az, R(sz, az), sz+1) into D;
12:       Sample random minibatch of transitions from D;
13:       if episode terminates at step z + 1 then
14:            Q(sz, az) := R(sz, az);
16:            Update Q(sz, az) by (56);
17:       end if
18:       Perform a gradient descent step by (60);
19:       Every \iota steps reset Qtarget (sz, az) := Q(sz, az);
20:       if z \geq 0 and R(sz, az) \leq 0, \forallz \in [z - 0, z] then
21:                f lag := 1;
22:       end if
23:       if z = |A| + 1 or f lag then
24:            Break;
25:       end if
26:       end for
27:     end for
28:     return A.

R(sz, az) is defined by,

\sum_{m}      \sum_{m}
R(sz, az) =  ℓi^{\prime}^{\prime} -      ℓi^{\prime} ,

where ℓ^{\prime}^{\prime} and ℓ^{\prime} indicate the total trade-off cost before and after executing ai z, respectively. Note that R(sz, az) is set to -\infty, if any one of constraints is not met for problem P.

The framework of training in stage 1 of PRE is the same as that except for environment, state, action and descriptions for action-value function and loss function are omitted, to avoid redundant descriptions. According to resulting action X (t), we can directly obtain scheduling decisions Y(t) at microarchitecture-level from resulting Am derived from PRE. For example, if the fully connected or convolution layer ri j (t) is scheduled to vehicle vk, then yi jk (t) is the scheduling decision on vehicle vk in Am. Algorithm 2 shows the details of DRL.

Note that there is one hidden layer with ℏ neurons, both for DRL and PRE. For the microarchitecture-level, the number of input neurons is |L|, and the number of output neurons is 2|L| in PRE. Thus, for microarchitecture-level, the number of multiplication operations is (|L|\cdotℏ+2|L|\cdotℏ), i.e., 3ℏ|L| in PRE. For the network-level, the number of input neurons is m, and the number of output neurons is m \cdotn. Thus, for network-level, the number of multiplication operations is (m \cdotℏ +m \cdotn \cdotℏ), i.e., (n + 1)mℏ. The number of multiplication operations for DRL is max{3ℏ|L|, (n + 1)mℏ}. Let mˆ = max{3ℏ|L|, (n + 1)mℏ}. This concludes that the algorithm DRL works in O(mˆ).

# VII. SIMULATION RESULTS

# A. Simulation Dataset and Platform

In this section, the performance of our proposed algorithms is evaluated on an integrated platform, as shown in Fig. 5. Specifically, the integrated platform is comprised of a desktop, Raspberry Pi, Eyeriss, open street map (OSM), simulation of urban mobility (SUMO) and network simulator NS-3.

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18, 2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

WU et al.: TWO-LEVEL SCHEDULING ALGORITHMS FOR DNN INFERENCE IN VEHICULAR NETWORKS

Fig. 5. The framework of simulations.

Fig. 7. The DNN models used in the simulations.

with point (23.054538^\circ N, 113.396124^\circ E) and end with point (23.045021^\circ N, 113.395245^\circ E) in simulations. Besides, all roads are double lanes, which are the same as the real road environment. Then, vehicle trajectories are generated by SUMO with “randomTrips.py” and the imported road topology. Here, the maximum speed is set to 40 \frac{km}{h} for each vehicle, which is the limited maximum speed in the road of the real environment. Besides, the acceleration and deceleration are set to 2.6 \frac{m}{s²} and 4.5 \frac{m}{s²}, respectively, for each vehicle, which are the default setting in SUMO.

Fig. 6. The road topology exploited in simulation.

Meanwhile, all vehicles drive in the same direction on the straight road, and the probability is the same for each turning direction for each vehicle on the crossings. After that, the generated vehicle trajectories are imported into NS-3. In NS-3, V2V communication is implemented by protocol 802.11p, which are widely applied in real transportation system.

# 3) DNN Inference Task Simulation:

In the transportation system, the applications for image recognition are widely applied, such as landmark recognition, traffic light classification, etc. Without loss of generality, four kinds of DNN inference tasks are selected to evaluate algorithm for heterogeneous task requests, where the corresponding DNN models are well known and widely adopted for image recognition applications. Fig. 7 shows the corresponding DNN models, including NiN [45], MobileNetV3-Small [46], RestNet18 [47] and VGG16 [48]. The input for DNN inference tasks is an image with 224 \times 224 \times 3 pixels and its data size is 147 KB. Besides, the data size of feedback for DNN inference tasks is set to 2 KB. The type of each task in a vehicular network is random one of the four kinds of DNN inference tasks.

# 4) Default Parameters:

The default parameters of simulations are the same as that in [5], [20], and [19]. The bandwidth Bz and transmit power p are set to 10 MHz and 27 dBm, respectively. Besides, the values of \alpha_{i} and \beta_{i} vary in [0, 1] and [0, \frac{1}{3}], respectively, to achieve the same order of magnitude between delay cost and energy cost. Note that \alpha_{i} is set to (1-\beta_{i})/3, for 1 \leq i \leq m, to represent the preference between response time and energy consumption for QoS of each task. According to the performance of DRL, the related parameters are detailed as follows. Both value network and target network have one hidden layer, which has 128 neurons. Besides, \gamma, \delta, \epsilon, \nu, 0 and mini-batch are set to 0.01, 0.7, 0.9, 2000, 200 and 128, respectively, both for DRL and for its pre-trained model.

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18,2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 24, NO. 9, SEPTEMBER 2023

PRE. The values for 3 are set to 3000 and 100 for PRE and DRL, respectively.

# B. Baseline Algorithms

To the best of our knowledge, the proposed algorithms in this paper are the first work to investigate two-level scheduling strategy for minimizing total trade-off cost in a vehicular network. Thus, no existing work can be used as direct baseline for comparisons, due to the challenges as shown in section II. Inspired by the state-of-the-arts in vehicular networks, we customize baseline algorithms for fair comparisons.

Inspired by [29], we customize a baseline algorithm based on greedy strategy, denoted as BA-G. Specifically, BA-G constructs a set S, which consists of all possible execution devices in a vehicular network. Both CPU and accelerator in a vehicle are regarded as execution device in BA-G. The network environment for two devices in a vehicle is same. Then, BA-G schedules each layer for each DNN inference task to execution devices, such that the trade-off cost is minimized in S.

Then, we also construct a baseline algorithm for task scheduling at microarchitecture-level. Inspired by [28], we customize a baseline algorithm based on greedy strategy, denoted as BA-GM, for scheduling DNN inference tasks at microarchitecture-level. Specifically, BA-GM greedily schedules each layer of tasks to CPU or accelerator with minimum total trade-off cost.

After that, we construct two baseline algorithm for task scheduling at network-level, by considering whether the task can be divided. Firstly, inspired by [5], we customize a baseline algorithm based on coalition game for indivisible task scheduling at network-level, which is denoted as BA-TN. Specifically, BA-TN randomly selects CPU or accelerator as the computing unit for the vehicle deployed with CPU and accelerator. Then, BA-TN regards tasks in a vehicle as a coalition. The profit of a coalition is the negative value of total trade-off cost for all tasks in the coalition. BA-TN traverses all coalitions once to form new coalition structure based on switch rule for each task, to maximize the system profit. Finally, BA-TN schedules DNN inference task according to the final coalition structure.

Then, inspired by [30], we customize a baseline algorithm, denoted as BA-GN, based on greedy strategy for divisible task scheduling at network-level. First, BA-GN randomly selects CPU or accelerator on a vehicle as its computing units. Then, BA-GN prefers to schedule each layer of each DNN inference task to the vehicle with minimum trade-off cost.

# C. Performance Comparison

Let ϱ and \rho be the number of failed tasks and the proportion of failed tasks, respectively. We say there is a failed task if and only if the task are unsuccessfully executed in simulations, due to data loss in communication and unsatisfied in constraints of problem P. Thus, \rho is defined as (\frac{ϱ}{m}) \cdot 100%.

Fig. 8 shows performance comparison among baseline algorithms BA-G, BA-GM, BA-TN, BA-GN and the proposed two algorithms GTA, DRL, in terms of total trade-off cost in a vehicular network for different values of \alpha_{i} for i \in M. For simplicity, in the figure, \alpha_{i} = \alpha for i \in M. In this simulation, n = 9, m = 10, Ekmax (t) = 30J, Timax (t) = 6.5s and qkmax = 24000 KB, for k \in N , i \in M. From Fig. 8, the proposed algorithms GTA and DRL outperform baseline algorithms BA-G, GA-GM, GA-TN, GA-GN for most cases. From Fig. 8, GTA can reduce the total trade-off cost by 12.31%, 15.74%, 4.77% and 9.57%, respectively. Meanwhile, DRL also can reduce the total trade-off cost by 30.61%, 33.32%, 24.64% and 28.44%, respectively.

For the case of \alpha_{i} = 0.9, compared with BA-G, GA-GM, GA-TN and GA-GN, GTA can successfully reduce total trade-off cost by 49.79%, 54.89%, 47.84% and 61.80%, respectively. Meanwhile, DRL can also reduce total trade-off cost by 52.60%, 57.41%, 50.76% and 63.94%, respectively.

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18,2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

WU et al.: TWO-LEVEL SCHEDULING ALGORITHMS FOR DNN INFERENCE IN VEHICULAR NETWORKS

# TABLE II

# PERFORMANCE COMPARISONS AMONG ALGORITHMS, IN TERMS OF PROPORTION OF FAILED TASKS FOR DIFFERENT MAXIMUM TOLERANT ENERGY CONSUMPTION AND STORAGE CAPACITY ON EACH VEHICLE

Fig. 10. Performance comparisons between BA-G, BA-GM, BA-TN, BA-GN and proposed GTA, DRL, in terms of total trade-off cost for different maximum allowed storage capacity on each vehicle.

Fig. 11. Performance comparisons between BA-G, BA-GM, BA-TN, BA-GN and proposed GTA, DRL, in terms of total trade-off cost for different maximum tolerant response time for each task.

and DRL perform better than baseline algorithms BA-G, BA-GM, BA-TN, BA-GN, in terms of total trade-off cost for all cases. For example, for the case of Ekmax (t) = 25J, compared with that for BA-G, BA-GM, BA-TN and BA-GN, GTA can reduce total trade-off cost by 20.68%, 25.26%, 16.06% and 19.60%, respectively. Besides, in this case, DRL can also reduce total trade-off cost by 33.90%, 37.73%, 30.06% and 37.06%, respectively. In addition, as Ekmax (t) increases from 5J to 15J, the general trend of total trade-off cost increases for all algorithms. This is because the corresponding proportion of failed tasks decreases, as shown in table II, which leads to that total trade-off cost decreases. From table II, \rho is smaller for DRL than that for other algorithms, and \rho is smaller for GTA than that for BA-G, BA-GM, BA-GN, for most cases. Especially for the cases of Ekmax (t) \geq 15J, the value of \rho is kept as 0% for GTA and DRL, while it is 1.11% and 2.22% for BA-GM and BA-GN. This can conclude that the proposed DRL and GTA outperform the baseline for most cases both in total trade-off cost and in the proportion of failed tasks.

Fig. 10 shows performance comparison among algorithms in terms of total trade-off cost for different maximum allowed storage capacity on each vehicle. In this simulation, n = 9, m = 10, \alpha_{i} = 0.5, Ekmax (t) = 30J, and qimax (t) = 24000 KB, for k \in N, i \in M. As shown in Fig. 11, DRL outperforms other algorithms for all cases, and GTA outperforms baseline algorithms for the cases of Timax (t) \geq 3s. For example, for the case of Timax (t) = 4.5s, compared with BA-G, BA-GM, BA-TN and BA-GN, GTA can reduce total trade-off cost by 17.85%, 23.06%, 15.12%

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18, 2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 24, NO. 9, SEPTEMBER 2023

# TABLE III

# RUNNING TIME COMPARISONS AMONG ALGORITHMS FOR DIFFERENT NUMBERS OF VEHICLES IN A VEHICULAR NETWORK

Fig. 12. Performance comparisons between BA-G, BA-GM, BA-TN, BA-GN and proposed GTA, DRL, in terms of total trade-off cost for different number of tasks.

by 14.72%, 20.72%, 13.06% and 22.81%, respectively. Meanwhile, in this case, DRL can also reduce total trade-off cost by 45.16%, 49.02%, 44.09% and 50.36%, respectively.

Fig. 13 shows performance comparison between the proposed GTA, DRL and baseline algorithms BA-G, BA-GM, BA-TN, BA-GN, in terms of total trade-off cost for different numbers of vehicles. In this simulation, m = 10, \alpha_{i} = 0.5, Ekmax (t) = 30J, Timax (t) = 6.5s, and qimax (t) = 24000 KB, for k \in N, i \in M. From Fig. 13, the total trade-off cost for algorithm DRL is obviously lower than that for baseline algorithms for all cases. Besides, total trade-off cost for GTA is lower than that for baseline algorithms for the cases of n \geq 6. For example, for the case of n = 15, in comparison to BA-G, BA-GM, BA-TN and BA-GN, DRL can reduce the total trade-off cost by 46.37%, 44.74%, 40.22% and 45.70%, respectively. Besides, GTA can also reduce the total trade-off cost by 17.85%, 23.06%, 15.65% and 21.69%, respectively. For the case of n = 3, total trade-off cost for GTA is slightly lower than that for BA-TN. This is because the proportion of failed tasks is 0% for GTA, while it is 1.11% for BA-TN in this case. The general trend of the total trade-off cost decreases, with increasing number of vehicle. This is because the total computing ability increases for a vehicular network, with the increasing number of vehicle. In addition, Fig. 13 also shows the standard deviation of different runs of simulations. From Fig. 13, the standard deviations for the proposed algorithms, especially for DRL, are smaller than that for baseline algorithms for most cases. This implies that the performance of the proposed algorithms is more stable, compared with baseline algorithms. Meanwhile, the standard deviations are not small for all algorithms. This is because the possibility to be scheduled is higher, when the value of Timax (t) increases. Thus, GTA can successfully reduce the total trade-off cost by scheduling tasks in two-level.

Fig. 12 shows performance comparison among different algorithms in terms of total trade-off cost for different number of DNN inference tasks. In this simulation, n = 9, \alpha_{i} = 0.5, Ekmax (t) = 30J, Timax (t) = 6.5s, and qimax (t) = 24000 KB, for k \in N, i \in M. From Fig. 12, the proposed GTA and DRL always perform better than four baseline algorithms for all cases. For example, for the case of m = 8, compared with that for BA-G, BA-GM, BA-TN and BA-GN, GTA can successfully reduce the total trade-off cost.

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18, 2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

WU et al.: TWO-LEVEL SCHEDULING ALGORITHMS FOR DNN INFERENCE IN VEHICULAR NETWORKS

all cases. This implies that the contributions at network-level are larger than that at microarchitecture-level, for the proposed algorithms. For example, for the case of that n = 9, GTA_{Net} and DRL_{Net} can successfully reduce the trade-off cost by 6.49% and 28.44%, compared with GTA_Micro and DRL_Micro, respectively.

# VIII. CONCLUSION

In this paper, we have formulated a novel two-level scheduling problem for DNN inference tasks in a vehicular network. The objective of the formulated problem is to minimize the total weighted sum of response time and energy consumption for all DNN inference tasks (i.e., total trade-off cost), under constraints of response time for each task, energy consumption for each vehicle, storage capacity for each vehicle, etc. The proof for the NP-hardness of the formulated problem has been also given. We have proposed GTA, a group transformation based algorithm, that exploits group transformation rule and greedy strategy. In addition, we have proposed DRL, a deep reinforcement learning based algorithm, that utilizes binding operation and training of two models based on deep reinforcement learning, to reduce total trade-off cost. A desktop, Raspberry Pi, Eyeriss, OSM, SUMO and NS-3 have been integrated as a evaluation platform to co-simulate the practical environment of a vehicular network. Extensive simulation results on the integrated platform show that the proposed DRL outperforms existing works BA-G, BA-GM, BA-TN and BA-GN for all cases, in terms of total trade-off cost.

The running time of DRL is relatively stable for different numbers of vehicles. The running time of GTA increases with the increasing number of vehicles. This is because the time complexity of GTA is positively associated with the number of vehicles, as shown in section VI-A. In conclusion, the proposed DRL outperforms the baseline algorithms for all cases, and the proposed GTA performs better than the baseline algorithms for most cases, in terms of total trade-off cost. Besides, the proposed DRL and GTA outperform the baseline algorithms for most cases, in terms of proportion of failed tasks.

Fig. 14 shows the contributions for each component of the proposed algorithms GTA and DRL, in terms of total trade-off cost for different numbers of vehicles in a vehicular network. The parameters in this simulation are the same as that of Fig. 13. Note that DRL and GTA jointly schedule inference tasks at microarchitecture-level and network-level. In order to show the contributions of different components for proposed algorithms, we compare GTA and DRL with the following strategies. (1) Strategy GTA_{Net}: GTA only schedules inference tasks at network-level, where each vehicle processes the tasks on CPU. (2) Strategy GTA_Micro: GTA only schedules inference tasks at microarchitecture-level, i.e., strategy GS in section VI-A. (3) Strategy DRL_{Net}: DRL only schedules inference tasks at network-level, where each vehicle processes the tasks on CPU. (4) Strategy DRL_Micro: DRL only schedules tasks at microarchitecture-level, i.e., each task is scheduled between CPU and accelerator in a vehicle. From Fig. 14, the trade-off costs for GTA_{Net} and DRL_{Net} are smaller than that for GTA_Micro and DRL_Micro, for n \leq 12. Meanwhile, the running time of DRL is smaller than that of BA-TN, for the cases of that n \geq 9. Besides, GTA and DRL run more slowly than BA-G, BA-GM and BA-GN for most cases. This is because, for each task, GTA iteratively traverses all task groups to form a stable group family. Besides, DRL includes deep neural network. These lead to time consuming. Meanwhile, BA-G, BA-GM and BA-GN are heuristic algorithms, their running time is relatively small. In addition, the running time is evaluated on a CPU. But, these algorithms, especially for DRL, can be directly accelerated by deploying on RSU with GPU. Thus, the increasing of running time is acceptable for GTA and DRL.

It is well known that there are many challenge problems both for machine learning training and for inference in transportation system, due to complex environment and characterizes of tasks. Exploring more efficient algorithms falls into our future works for security, incentive mechanisms, more complex road map, etc., to achieve machine learning training and inference with high quality of services. For example, blockchain or encryption techniques may be exploited to guarantee the security of model sharing. Besides, auction or coalition game may be exploited to encourage vehicles to share their computing resources. For machine learning training, the algorithms for jointly task scheduling and noise reduction in data will be considered in the future works. For machine learning inference, developing distributed algorithms for two-level task scheduling is one of our future works in vehicular networks.

# REFERENCES

- [1] Z. Ning et al., “Intelligent edge computing in Internet of Vehicles: A joint computation offloading and caching solution,” IEEE Trans. Intell. Transp. Syst., vol. 22, no. 4, pp. 2212-2225, Apr. 2021.
- [2] Z. Ouyang, J. Niu, Y. Liu, and X. Liu, “An ensemble learning-based vehicle steering detector using smartphones,” IEEE Trans. Intell. Transp. Syst., vol. 21, no. 5, pp. 1964-1975, May 2019.
- [3] J. Bi, H. Yuan, K. Zhang, and M. Zhou, “Energy-minimized partial computation offloading for delay-sensitive applications in heterogeneous edge networks,” IEEE Trans. Emerg. Topics Comput., vol. 10, no. 4, pp. 1941-1954, Dec. 2022.

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18,2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

# IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 24, NO. 9, SEPTEMBER 2023

# References

1. H. Yuan, J. Bi, J. Zhang, and M. Zhou, “Energy consumption and performance optimized task scheduling in distributed data centers,” IEEE Trans. Syst., Man, Cybern. Syst., vol. 52, no. 9, pp. 5506-5517, Sep. 2022.
2. Y. Wu, J. Wu, L. Chen, J. Yan, and Y. Han, “Load balance guaranteed vehicle-to-vehicle computation offloading for min-max fairness in VANETs,” IEEE Trans. Intell. Transp. Syst., vol. 23, no. 8, pp. 11994-12013, Aug. 2022.
3. E. E. Agent. Annual European Union Greenhouse Gas Inventory 1990-2018 and Inventory Report 2020. Submission to the UNFCCC Secretariat. Accessed: Dec. 12, 2020. [Online]. Available: https://www.eea.europa.eu//\frac{publications}{european}-union-greenhouse-gas-inventory-2020
4. Communication From the Commission to the European Parliament, the Council, the European Economic and Social Committee and the Committee of the Regions: A European Strategy for Low-Emission Mobility, European Commission, Brussels, Belgium, 2016.
5. A. Meneses-Viveros, M. Paredes-López, E. Hernández-Rubio, and I. Gitler, “Energy consumption model in multicore architectures with variable frequency,” J. Supercomput., vol. 77, no. 3, pp. 2458-2485, Mar. 2021.
6. Z. Liao, J. Fu, and J. Wang, “Ameliorate performance of memristor-based ANNs in edge computing,” IEEE Trans. Comput., vol. 70, no. 8, pp. 1299-1310, Aug. 2021.
7. D. Xu et al., “Edge intelligence: Architectures, challenges, and applications,” 2020, arXiv:2003.12172.
8. X. Wang, Y. Han, V. C. M. Leung, D. Niyato, X. Yan, and X. Chen, “Convergence of edge computing and deep learning: A comprehensive survey,” IEEE Commun. Surveys Tuts., vol. 22, no. 2, pp. 869-904, 2nd Quart., 2020.
9. M. Milosevic, M. Z. Bjelica, T. Maruna, and N. Teslic, “Software platform for heterogeneous in-vehicle environments,” IEEE Trans. Consum. Electron., vol. 64, no. 2, pp. 213-221, May 2018.
10. M. Z. Bjelica and Z. Lukac, “Central vehicle computer design: Software taking over,” IEEE Consum. Electron. Mag., vol. 8, no. 6, pp. 84-90, Nov. 2019.
11. S. Baidya, Y.-J. Ku, H. Zhao, J. Zhao, and S. Dey, “Vehicular and edge computing for emerging connected and autonomous vehicle applications,” in Proc. \frac{ACM}{IEEE} Design Autom. Conf., Oct. 2020, pp. 1-6.
12. M. Song et al., “In-situ AI: Towards autonomous and incremental deep learning for IoT systems,” in Proc. IEEE Int. Symp. High Perform. Comput. Archit. (HPCA), Feb. 2018, pp. 92-103.
13. J. Zhang and K. B. Letaief, “Mobile edge intelligence and computing for the Internet of Vehicles,” Proc. IEEE, vol. 108, no. 2, pp. 246-261, Feb. 2019.
14. T. Mohammed, C. Joe-Wong, R. Babbar, and M. Di Francesco, “Distributed inference acceleration with adaptive DNN partitioning and offloading,” in Proc. IEEE INFOCOM Conf. Comput. Commun., Jul. 2020, pp. 854-863.
15. K. Xiong, S. Leng, C. Huang, C. Yuen, and Y. L. Guan, “Intelligent task offloading for heterogeneous V2X communications,” IEEE Trans. Intell. Transp. Syst., vol. 22, no. 4, pp. 2226-2238, Apr. 2021.
16. J. Zhao, X. Sun, Q. Li, and X. Ma, “Edge caching and computation management for real-time Internet of Vehicles: An online and distributed approach,” IEEE Trans. Intell. Transp. Syst., vol. 22, no. 4, pp. 2183-2197, Apr. 2021.
17. Y. Wu, J. Wu, L. Chen, G. Zhou, and J. Yan, “Fog computing model and efficient algorithms for directional vehicle mobility in vehicular network,” IEEE Trans. Intell. Transp. Syst., vol. 22, no. 5, pp. 2599-2614, May 2020.
18. Y. H. Chen, T. Krishna, J. S. Emer, and V. Sze, “Eyeriss: An energy-efficient reconfigurable accelerator for deep convolutional neural networks,” IEEE J. Solid-State Circuits, vol. 52, no. 1, pp. 127-138, May 2017.
19. R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction. Cambridge, MA, USA: MIT Press, 2018.
20. Y. Xiang and H. Kim, “Pipelined data-parallel \frac{CPU}{GPU} scheduling for multi-DNN real-time inference,” in Proc. IEEE Real-Time Syst. Symp. (RTSS), Dec. 2019, pp. 392-405.
21. B. Zhang, H. Zeng, and V. Prasanna, “Hardware acceleration of large scale GCN inference,” in Proc. IEEE 31st Int. Conf. Appl.-Specific Syst., Archit. Processors (ASAP), Jul. 2020, pp. 61-68.
22. X. He et al., “Enabling energy-efficient DNN training on hybrid GPU-FPGA accelerators,” in Proc. ACM Int. Conf. Supercomputing, Jun. 2021, pp. 227-241.
23. Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18, 2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

WU et al.: TWO-LEVEL SCHEDULING ALGORITHMS FOR DNN INFERENCE IN VEHICULAR NETWORKS

Yalan Wu received the B.Sc. and Ph.D. degrees from the Guangdong University of Technology (GDUT), China, in 2016 and 2021, respectively. In 2017, she was an Intern Student with Nanyang Technological University, Singapore, for the joint project of high-performance architecture. She is currently a Post-Doctoral Researcher with the School of Integrated Circuits, GDUT, and a Visiting Scholar with the School of Computer Science and Engineering, Nanyang Technological University. Her research interests include vehicular networks, mobile computing, and high-performance architecture.

Long Chen received the B.Eng. degree in computer science from Anhui University, Hefei, Anhui, China, in 2011, and the Ph.D. degree from the School of Computer Science and Technology, USTC, in 2016. He is currently an Associate Professor with the School of Computer Science and Technology, Guangdong University of Technology. He has published several journal articles in IEEE TRANSACTIONS ON SERVICES COMPUTING and IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY. His research interests include cognitive radio networks, network economics, and mobile computing.

Jigang Wu (Member, IEEE) received the B.Sc. degree from Lanzhou University and the Ph.D. degree from the University of Science and Technology of China. He was with Nanyang Technological University, Singapore, from 2000 to 2010. He was a Tianjin Distinguished Professor with Tianjin Polytechnic University, China, from 2010 to 2015. He is currently a Distinguished Professor with the School of Computer Science and Technology, Guangdong University of Technology. He has published more than 300 articles. His research interests include network computing and machine learning.

Siew Kei Lam (Senior Member, IEEE) received the B.A.Sc., M.Eng., and Ph.D. degrees from Nanyang Technological University (NTU), Singapore. He was a Visiting Research Fellow with the Imperial College of London, the University of Warwick, and RWTH Aachen, Germany. He is currently an Associate Professor with the School of Computer Science and Engineering (SCSE), NTU. His research investigates custom computing techniques to meet the challenging demands for performance, energy-efficiency, cost, reliability, and security in edge intelligence. His research group develops novel design methodologies, domain-specific architectures, and architecture-aware algorithmic optimizations that will enable complex applications to run on edge devices to provide timely operations, while being robust to faults and security attacks. He has published more than 150 international refereed journals and conferences in these areas. He is an Associate Editor of the IET Circuits, Devices and Systems journal.

Authorized licensed use limited to: Guangdong Univ of Tech. Downloaded on December 18, 2023 at 11:50:45 UTC from IEEE Xplore. Restrictions apply.

