#include <ros/ros.h>

#include <tf/transform_listener.h>

#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose2D.h>

#include <Eigen/Dense>

#include <gazebo_msgs/SpawnModel.h>
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/ModelStates.h>

#include <eigen_conversions/eigen_msg.h>

#include <fstream>

#include "ilqg/ilqg_differential_mobile.h"


Eigen::Isometry3d robot_transform;
bool isFirstUpdateDone = false;
Eigen::Vector3d robot_state;

void gazeboModelCallback(const gazebo_msgs::ModelStatesConstPtr& msg)
{
  for(int i=0; i<msg->name.size(); i++)
  {
    if (msg->name[i] == "mobile_base")
    {
      tf::poseMsgToEigen(msg->pose[i], robot_transform);
    }
    //std::cout << robot_transform.translation() << std::endl;
  }
}

void slamPoseCallback(const geometry_msgs::Pose2DConstPtr& msg)
{
  robot_state(0) = msg->x;
  robot_state(1) = msg->y;
  robot_state(2) = msg->theta;

  isFirstUpdateDone = true;
}

int main(int argc, char **argv)
{
  const size_t CONTROL_HORIZON = 70;
  const double CONTROL_FREQ = 0.1;


  ros::init(argc, argv, "husky_control_ilqg");
  ros::NodeHandle nh("~");

  ros::Publisher vel_cmd_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel",2);
  ros::Subscriber slam_pose_sub = nh.subscribe("/pose2d", 1, slamPoseCallback);

  //  ros::Subscriber
  /*
  ros::Subscriber gazebo_model_sub = nh.subscribe("/gazebo/model_states", 1, gazeboModelCallback);
  ros::ServiceClient gazebo_set_client = nh.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");

  gazebo_msgs::SetModelStateRequest req;
  gazebo_msgs::SetModelStateResponse res;



  req.model_state.model_name = "mobile_base";
  req.model_state.pose.position.x = .0;
  req.model_state.pose.position.y = .0;
  req.model_state.pose.position.z = 0.133;
  req.model_state.pose.orientation.z = 0.707;
  req.model_state.pose.orientation.w = 0.707;


  //req.model_state
  gazebo_set_client.call(req,res);
*/

  ROS_INFO("Waiting to receive first pose");

  while (ros::ok() && !isFirstUpdateDone)
  {
    ros::spinOnce();
  }

  ROS_INFO("First pose:");
  std::cout << robot_state << std::endl;

  iLQGDifferentialMobile dm(0.545, CONTROL_FREQ);
  Matrix<double, 5, 1> x0, xd, xt;
  MatrixXd u0(2, CONTROL_HORIZON);
  MatrixXd u_limit(2,2);


  x0.setZero();
  u0.setZero();
  x0.head<3>() = robot_state;
  
  xd << -0.5, -0.5, M_PI/2 , 0, 0;


  u_limit << -0.3, 0.3,
      -0.3, 0.3;

  dm.init(x0, u0, CONTROL_HORIZON);
  dm.setDesiredState(xd);
  dm.setInputConstraint(u_limit);
  dm.setVerboseLevel(dm.vbl_info);
  dm.plan();

  //dm.plan();
  //dm.plan();
  //dm.plan();

  auto &plan = dm.getPlannedStateMatrix();

  Eigen::MatrixXd plan_ori;
  //ROS_INFO("planned size = %d %d", plan.rows(), plan.cols());
  plan_ori.resize(plan.rows(), plan.cols());
  //ROS_INFO("resized");
  plan_ori = plan;
  //ROS_INFO("copied");

  ros::Rate r(1 / CONTROL_FREQ);
  size_t time_tick = 0;

  /*
  ROS_INFO ("Wait for robot landing");
  for(int i=0; i<1 / CONTROL_FREQ * 3; i++)
  {

    r.sleep();
  }
  ROS_INFO ("Done");
*/
  ofstream file_log("/home/dyros/ilqg_log_data.txt");

  file_log << "x1\tx2\tx3\tx4\tx5\tr1\tr2\tr3\tr4\tr5" << endl;
  tf::TransformListener listener;
  geometry_msgs::Twist msg;
  double x_angle, y_angle;
  double z_angle = 0.0;
  tf::Quaternion quat;
  while (ros::ok())
  {
    tf::StampedTransform transform;
    try
    {
      listener.lookupTransform("/map", "/base_link", ros::Time(0), transform);
      transform.getBasis().getRPY(x_angle, y_angle, z_angle);
    }
    catch (tf::TransformException ex)
    {
      ROS_ERROR("%s", ex.what());
    }

    //ROS_INFO("start");
    //		double gain_w = 1.2;
    double gain_w = 1.2;
    double vl = plan(3,time_tick);
    double vr = plan(4,time_tick);
    double v = (vr+vl)/2.;
    double w = (vr-vl)/0.545 ;

    //ROS_INFO("start2");
    msg.linear.x = v;
    msg.angular.z = w;
    vel_cmd_pub.publish(msg);


    //ROS_INFO("start3");
    xt(0) = transform.getOrigin().x(); //robot_state(0);
    xt(1) = transform.getOrigin().y(); //robot_state(1);
    xt(2) = z_angle;
    xt(3) = vl;
    xt(4) = vr;
    //ROS_INFO("replan");
    dm.replan(time_tick,xt);

    //std::cout << "Diffs" << std::e  ndl;
    //std::cout << xt << std::endl;
    //std::cout << plan.col(time_tick) << std::endl;
    //std::cout << xt-plan.col(time_tick) << std::endl;
    file_log << xt(0) << '\t' << xt(1) << '\t' << xt(2) << '\t'<< xt(3) << '\t' << xt(4) << '\t'
             << plan_ori.col(time_tick)(0) << '\t'
             << plan_ori.col(time_tick)(1) << '\t'
             << plan_ori.col(time_tick)(2) << '\t'
             << plan_ori.col(time_tick)(3) << '\t'
             << plan_ori.col(time_tick)(4) << endl;

    time_tick ++;
    if(time_tick == CONTROL_HORIZON)
    {
      break;
    }

    r.sleep();
    ros::spinOnce();

    // Vector3d euler_angle;

    // euler_angle = robot_transform.linear().eulerAngles(2, 1, 0);
    //cout << robot_transform.linear() << endl<< endl;
    //cout << euler_angle << endl;

  }

  r.sleep();
  msg.linear.x = 0;
  msg.angular.z = 0;
  vel_cmd_pub.publish(msg);
  r.sleep();
}

