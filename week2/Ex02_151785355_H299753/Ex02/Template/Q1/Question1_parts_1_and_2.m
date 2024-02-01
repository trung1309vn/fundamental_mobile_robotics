%% made for MoRo course, 
% Exercise 2: Kalman filter, 1 D robot
% Reza Ghabcheloo, January 2024
% Complete the places denoted by "???"

x0 = 0; % true robot state
v = 0;% true robot speed and speed command 
P0 = 0.01^2; % initial belief uncertainty
pd_x = makedist('Normal','mu',x0,'sigma',sqrt(P0)); % robot position initial belief
xhat0 = random(pd_x);

RobotModel.A =  1;
RobotModel.H = -1;
RobotModel.Q = 0.1^2;
RobotModel.R = 0.03^2;

pd_dis_v = makedist('Normal','mu',0,'sigma',sqrt(RobotModel.Q)); % prediction uncertainty
pd_noise_d = makedist('Normal','mu',0,'sigma',sqrt(RobotModel.R)); % measurement uncertainty 

Ts = 0.05;
t_vec = 0:Ts:4; % simulation time trajectory
N = length(t_vec);

v_vec = [linspace(0,1,10) ones(1,20) linspace(1,-1,20) -ones(1,20) linspace(-1,0,10) zeros(1,N-80)]; % control input trajectory

%% In this part of the code, we only simulate the robot with control input v_vec and make prediction.
% that means, no measurements are available in this part

x_vec = zeros(1,N); % memory allocation to store robot true trajectory
xhat_vec = zeros(1,N); % memory allocation to store estimated robot trajectory
P_vec = zeros(1,N); % memory allocation to store estimated robot trajectory covariance

% initialization 
x_vec(1) = x0;   % store true robot trajectory
xhat_vec(1) = xhat0; % store expected predicted robot trajectory
P_vec(1) = P0; % store covariance of predicted robot trajectory

for k = 1:N-1
    % simulating the true robot

    x_vec(k+1) = step(x_vec(k), v_vec(k), pd_dis_v, RobotModel); 
    
    % prediction
    xhat_vec(k+1) = predict_model(xhat_vec(k), v_vec(k), RobotModel);
    P_vec(k+1) = predict_COV(P_vec(k), RobotModel);
end

figure(1)
subplot(2,1,1)
plot(t_vec,v_vec,'.-')
subplot(2,1,2)
plot(t_vec,x_vec,'.-',t_vec,xhat_vec,'.-',...
     t_vec,xhat_vec + sqrt(P_vec),'.-', t_vec,xhat_vec -sqrt(P_vec),'.-')
legend('true robot','expected','+1\sigma','-1\sigma')
    
%% In this part of the code, we simulate the robot with control input v_vec, and at each
% step, predict and update. That is, we have access to v (odometry) and z (sensor measurment)

x_vec = zeros(1,N); % memory allocation to store robot true trajectory

xhat_plus_vec = zeros(1,N); % memory allocation to store estimated robot trajectory, after update
P_plus_vec = zeros(1,N); % memory allocation to store estimated robot trajectory covariance, after update

xhat_neg_vec = zeros(1,N); % memory allocation to store estimated robot trajectory, before update
P_neg_vec = zeros(1,N); % memory allocation to store estimated robot trajectory covariance, before update

x_vec(1) = x0;
xhat_plus_vec(1) = xhat0;
P_plus_vec(1) = P0;

for k = 1:N-1
    % simulating the true robot 1 step forward in time
    x_vec(k+1)= step(x_vec(k), v_vec(k), pd_dis_v, RobotModel);

    % prediction
    xhat_neg_vec(k+1)= predict_model(xhat_plus_vec(k), v_vec(k), RobotModel);
    P_neg_vec(k+1) = predict_COV(P_plus_vec(k), RobotModel);
    
    % measure and update
    d = sense(x_vec(k+1), pd_noise_d, RobotModel);
    d_hat = sense_model(xhat_neg_vec(k+1), RobotModel);
    % Get Kalman gain for filtering 
    K = KalmanGain(P_neg_vec(k+1), RobotModel);
    I = 1
    P_plus_vec(k+1) = (I - K*RobotModel.H)*P_neg_vec(k+1);
    xhat_plus_vec(k+1) = xhat_neg_vec(k+1) + K*(d - d_hat);
        
end

figure(2)
plot(t_vec,x_vec,'.-',t_vec,xhat_neg_vec,'.-',...
    t_vec,xhat_neg_vec + sqrt(P_neg_vec),'.-', t_vec,xhat_neg_vec -sqrt(P_neg_vec),'.-',...
    t_vec,xhat_plus_vec + sqrt(P_plus_vec),'.-', t_vec,xhat_plus_vec -sqrt(P_plus_vec),'.-')
legend('true robot','expectation','+1\sigma before update','-1\sigma','+1\sigma after update','-1\sigma')
    
    
matlab comment code block
return

%% you may want to define following functions to make your code cleaner, and easier to debug

%% true robot and sensor simulation
function x = step(x, u, pd_dis_v, RobotModel)
% move robot forward in time
    x = RobotModel.A*x + u + random(pd_dis_v);
end

function d = sense(x,pd_noise_d, RobotModel)
% get a measurement when the robot is at x 
    d = 10 + RobotModel.H*x + random(pd_noise_d);
end

%% prediction and update functions

function x = predict_model(x, u, RobotModel)
% state prediction model
    x = RobotModel.A*x + u
end

function P = predict_COV(P,RobotModel)
% covariance prediction model
    P = RobotModel.A*P*RobotModel.A' + RobotModel.Q
end

function d = sense_model(x, RobotModel)
% sensor model: what the sensor would measure if the robot is at x
    d = 10 + RobotModel.H*x
end

function K = KalmanGain(P,RobotModel)
    K = P*RobotModel.H'/(RobotModel.H*P*RobotModel.H'+RobotModel.R);
end
