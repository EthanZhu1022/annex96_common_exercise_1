# COMP9991 Presentation Script

Target length: about 15 to 20 minutes.

Style: simple spoken English. Do not read every number on the slides. Use the numbers only to support the main story.

## Slide 1

Good morning everyone. My presentation is about multi-agent reinforcement learning for building energy control in CityLearn. The main idea is to control many houses together, so the whole group can follow a target electricity load more closely.

## Slide 2

This is the structure of my talk. I will first explain the background and the CityLearn task, then introduce the learning methods and communication designs, and finally discuss the results and future work. The key question is simple: which kind of coordination helps buildings work better as a group?

## Slide 3

The motivation comes from renewable energy and electrification. More homes now have solar panels, batteries, and heating or cooling systems that can be controlled. If we coordinate these homes, we can move electricity use away from difficult periods and make the total demand easier for the grid to manage.

## Slide 4

CityLearn is the simulation platform used in this project. It gives us a standard way to test building energy control methods. In this environment, each building has flexible devices like batteries and HVAC, and the controller can choose actions over time to improve the whole community performance.

## Slide 5

The benchmark is Annex 96 Common Exercise 1. It has 25 single-family homes, and the goal is for all homes together to follow a district reference load. Since there is no physical network map, I group buildings using simple flexibility features, mainly battery size and HVAC capacity.

## Slide 6

Here I define the two main load tracking metrics. NMBE tells us whether the controller is usually above or below the target. CV-RMSE tells us the size of the tracking error overall. For both metrics, lower is better, and comfort violations should also stay low.

## Slide 7

This slide shows the reward function used during training. In simple words, the agent gets punished when the total load is far from the target, and also when indoor temperature leaves the comfort range. So the controller needs to balance two things: following the load target and keeping homes comfortable.

## Slide 8

This slide shows the data split and the rule-based baseline. Vermont uses January for training and February for testing, while Texas is planned for the next stage. The rule-based controller is simple and transparent, but the errors are very high, so it is not enough for this task.

## Slide 9

Following the previous slide, I have already shown the RBC baseline results for Texas and Vermont. RBC means rule-based control. It does not learn from data. It just follows fixed if-then rules, for example charging or discharging the battery based on the current load situation. In this project, these rules come from CityLearn's `BasicBatteryRBC`, and the rule logic is written in `citylearn/agents/rbc.py` as a fixed hour-based action map.

The role of RBC is to give us a simple reference point. If a learning method cannot beat RBC, then the learning method is not very useful. In my results, RBC has large tracking errors, so it shows that fixed rules are too simple for this task. I focus on Vermont first because its target is less flat, so the difference between methods is easier to observe.

## Slide 10

Before introducing the algorithms, I want to explain on-policy and off-policy learning. On-policy methods learn from data collected by the current policy, which is usually stable but needs more samples. Off-policy methods reuse old data, which can be more efficient, but sometimes less stable.

## Slide 11

PPO is an on-policy method, and SAC is an off-policy method. PPO is usually stable and easier to debug. SAC can learn from a replay buffer and explores more, but in this task it can be harder to balance tracking and comfort.

## Slide 12

This table compares RBC, independent PPO, and independent SAC. RBC performs much worse than the learning methods. Independent PPO gives a strong balance, especially on comfort, while SAC has good CV-RMSE but much worse comfort and bias.

## Slide 13

Now I will move from single-agent methods to multi-agent reinforcement learning, or MARL. Here I use MAPPO, which is a stable on-policy algorithm and a useful backbone for the later research.

MAPPO is based on the CTDE idea, which means centralized training and decentralized execution. During training, the critic can use global information from all buildings, so it can judge the whole portfolio better. But during execution, each building only needs its own local observation to choose an action.

In the standard MAPPO setting here, I use one shared actor network and one shared critic network for the 25 buildings. The actor is the decision network. Each building sends its own observation, such as load, temperature, battery state, and time information, to the same actor network. The actor then outputs that building's next action, for example battery or HVAC control.

The critic is different. It is only used during training. It receives more global information, such as the joint state of all buildings, and outputs a value estimate. This value tells the algorithm whether the current decisions are good for the whole portfolio. So in simple words, the actor chooses actions, and the critic teaches the actor how good those actions are.

## Slide 14

Here I compare standard MAPPO with independent PPO and SAC. Standard MAPPO is not automatically better. In this result, independent PPO performs better, which shows that simply using a multi-agent method is not enough. The structure of the agents is very important.

## Slide 15

To improve the structure, I use K-means grouping. Buildings with similar flexibility are put into the same group, and buildings in the same group share one actor network. This reduces complexity and also respects the fact that not all houses behave the same way. Here, I build a two-dimensional feature for each building using HVAC capacity and battery capacity.

## Slide 16

This result shows that grouping helps. Grouped MAPPO reduces both CV-RMSE and NMBE compared with standard MAPPO, and comfort is also slightly better. So selective parameter sharing is a useful design choice for this building portfolio.

## Slide 17

After grouping, the next question is communication. Without communication, each building only uses its own encoded information, so it does not know what other buildings are doing. But the target is measured at the portfolio level, so one building's action can affect the whole group. This is why I add a communication layer before the final action decision.

The basic process is shown on this slide. First, each building takes its own observation. This observation is passed through an MLP encoder, so it becomes a hidden feature.

Then this hidden feature is used as the message that can be shared with other buildings. The communication layer receives messages from selected buildings and updates each building's hidden feature. Finally, the updated hidden feature goes into the action head, and the action head outputs the next control action for that building.

So the communication is not directly sending raw data like temperature or load. It sends a learned hidden representation, which is more compact and easier for the neural network to use.


## Slide 18

They are the communication modules that I implemented and compared in this project: CommNet, weighted communication, PowerNet, GAT, TarMAC, and DIAL. Each one uses a different way to decide what information should be shared between buildings. In the next few slides, I will analyze them one by one.

## Slide 19

CommNet is the simplest communication method here. Each agent receives the average hidden feature from other agents in its group. This is easy to implement, but it may mix useful information with noise, because every message is treated almost equally.

## Slide 20

Weighted communication is more controlled. It gives a higher weight to messages from similar buildings and a lower weight to messages from other groups. The results show that the weights matter a lot, because other-group information can either help the portfolio or become noise.

## Slide 21

This table shows that the communication weight has a clear effect on performance. The default weighted setting gives the best primary rank and the lowest comfort violation in this group, but the other two weight settings are worse in different ways.

The low ranks of CommNet and Comm v2 also show that simple communication is not enough. CommNet only uses same-group mean communication, and Comm v2 adds inter-group communication, but both still use a relatively simple message aggregation. When outside-group information is added without good filtering, it can bring noise, so the performance becomes worse.

Another important point is that giving more weight to the same group is not always better. The 0.90 and 0.10 setting gives more weight to the same group than the default setting, but it does not perform better. So the key is to find a suitable balance between same-group information and other-group information.

## Slide 22

From the weighted communication results, we can see that learning or choosing good message weights is very important. So I introduce TarMAC and GAT, which use attention to make the message weights more selective.

The basic idea of attention is to create three values from each hidden feature: query, key, and value. The query means what this building is looking for. The key means what information another building can provide. The value is the actual message content. By comparing query and key, the model calculates an attention weight, and then uses this weight to combine the value messages.

TarMAC and GAT use this attention idea, meaning the model learns which agents to listen to more. This is useful because not all buildings are equally important at every time. However, the current TarMAC version still lacks the stronger filtering used in PowerNet, so there is room to improve it. The main difference for GAT is that it needs a graph structure first, and in this project I build that graph from the K-means grouping.

## Slide 23

DIAL is another communication idea. It lets agents send differentiable messages during training, so learning can pass through the message channel. In this project, I use a sigmoid function during training to imitate a discrete communication signal, instead of sending a hard zero-or-one message directly. But the original DIAL idea was designed more for discrete-action settings, and in this continuous control task it does not perform very well.

## Slide 24

This slide gives a general comparison of the communication methods. TarMAC does not perform as well as expected here, so attention-based communication still needs more research and optimization in this task.

But we can also see that one method, PowerNet Global, performs very well. It gives the best overall result in this comparison. So next, I will explain what PowerNet does differently.

## Slide 25

PowerNet mainly uses feature concatenation, or concat. It concatenates the building's own hidden feature with the received message, and then passes this combined feature through another network.

The advantage is that local information is not directly overwritten by messages from other buildings. The model can still use shared information, but it also keeps the original local feature clearly separated.

## Slide 26

This is the original ranking table. The important point is not every row, but the pattern. PowerNet Global has the best overall rank, weighted default has the best primary rank, and the rule-based baseline is clearly the worst. So learning and communication both matter.

## Slide 27

These daily plots show the best overall run, PowerNet Global, across the February test month. The average NMBE is close to zero, which means the controller is not strongly biased upward or downward. But the daily CV-RMSE and comfort still vary a lot, so the controller is not equally good every day.

## Slide 28

This load tracking plot shows the full test month. The orange line is the learned controller, the blue line is the no-storage baseline, and the red dashed line is the target. The controller can shift the load and reduce some peaks, but it sometimes reacts too strongly and creates deep drops.

## Slide 29

This slide shows building temperatures during the first week. Many homes stay close to the comfort band, but some homes go above or below it for noticeable periods. This tells us that comfort is still a hard constraint, not just a small side issue.

## Slide 30

The full-month temperature plot makes the comfort issue clearer. Some buildings are mostly stable, while others have frequent violations. This means future work should not only improve the portfolio average, but also reduce bad outcomes for the worst buildings.

## Slide 31

This slide summarizes the primary metrics across selected experiments. The rule-based baseline is much worse than all learning methods. Among learning methods, PowerNet Global has the lowest CV-RMSE, while weighted default has a very strong balance between tracking and comfort.

## Slide 32

These are secondary metrics. They show that a method can be good on the main tracking objective but still have weaknesses in peak demand or ramping. These secondary metrics are not included in the training reward at this stage, but they help show which algorithms are naturally more stable and provide useful guidance for future research.

## Slide 33

Here I summarize the communication analysis. PowerNet Global works well because it keeps local information and adds structured shared information. Weighted communication shows that message weights matter. TarMAC is promising because it learns attention, while DIAL is less suitable for this continuous-control setting.

## Slide 34

To conclude, structural choices are very important in this task. Grouping and communication design both influence load tracking, comfort, and peak metrics. In the current results, PowerNet Global gives the best overall balance, while the weighted default method performs best on the primary metrics, although it increases peak demand.

There are also some limitations. The current encoder is still a simple MLP, so it mainly uses the current observation and does not explicitly remember past states. In the future, this encoder can be upgraded to an LSTM, so the model can use historical information and better capture time-dependent building behavior. The communication design also needs more work, because static grouping may limit adaptivity.

For future work, I will extend the experiments to Texas, run multi-seed evaluation, combine PowerNet-style filtering with TarMAC-style attention, and test stronger critic designs and other algorithms such as MASAC or value-based methods.

## Slide 35

This slide lists the first part of the references. I will not go through them one by one, but they support the background on CityLearn, reinforcement learning, building energy control, PPO, SAC, and grouping.

## Slide 36

This slide lists the second part of the references. These papers support the multi-agent methods and communication designs, such as PowerNet, CommNet, TarMAC, DIAL, and value-decomposition methods.

## Slide 37

That is the end of my presentation. Thank you for listening. I am happy to take questions.
