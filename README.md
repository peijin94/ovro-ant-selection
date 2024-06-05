# Select subset of antenna


Previous (recent) fast vis configuration and imaging result:


From observation of full set, select 48 to do imaging, the following is an example: red is core, blue is extended, green is 48 selected
<img width="771" alt="image" src="https://github.com/peijin94/ovro-ant-selection/assets/11596456/59fa8ea0-528f-402c-aa90-c88515852d60">
<img width="1087" alt="image" src="https://github.com/peijin94/ovro-ant-selection/assets/11596456/ff999b5b-dac7-473d-a234-1245a30fe9cc">
SNR 37.66 bA:88 bmaj 8.94

## Method

pick 48 from slow vis ms

(1) Evaluate the imaging results with: SNR, Area of beam at 0.3 peak, bmaj

(2) Replace some antennas (8 ants in this case)

(3) Re-evaluate, and decide to accept or reject the replace

(4) score the replaced antenna based on the evaluation

(5) Replace antenna randomly with weighting based on antenna score

repeat 3-5 until satisfied


the scores (yellow is higher):
<img width="678" alt="image" src="https://github.com/peijin94/ovro-ant-selection/assets/11596456/1de80541-53a6-46b4-b194-624cdeaadd08">


## Result
<img width="751" alt="image" src="https://github.com/peijin94/ovro-ant-selection/assets/11596456/c0fd5028-2ad3-4fb5-9531-2cdb6670942e">

<img width="1105" alt="image" src="https://github.com/peijin94/ovro-ant-selection/assets/11596456/2456c1eb-b2b6-4a79-b487-75793752a506">


SNR:41.80 bA:101 bmaj:8.85

(10% SNR improvement with smaller bmaj, but larger bA)

or much better SNR with slightly worse beam:
<img width="671" alt="image" src="https://github.com/peijin94/ovro-ant-selection/assets/11596456/b6bf8766-dd8a-464a-9c5a-37782d6f3c35">
<img width="1084" alt="image" src="https://github.com/peijin94/ovro-ant-selection/assets/11596456/4773a97b-aebb-4ca9-b2a8-ff6dab83171c">
 SNR:51.37      bArea:137.25   bmaj:10.10
(35% SNR improvement)


current configuration recomendation:
```
60;25;191;314;51;209;124;303;197;282;351;284;221;95;1;24;120;46;38;53;251;55;101;42;152;49;223;90;252;346;48;317;5;122;349;35;189;123;348;304;0;9;43;93;23;170;165;157
```
