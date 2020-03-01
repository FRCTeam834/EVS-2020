/*----------------------------------------------------------------------------*/
/* Copyright (c) 2019 FIRST. All Rights Reserved.                             */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

package frc.robot.subsystems;

import java.util.ArrayList;

import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.wpilibj2.command.SubsystemBase;

/**
 * Data structure for returned arrays is as follows: 
 * - Each column (first value) represents a space to store object values. 
 * The data structure of the rows is
 * as follows: - 0 or -1, with -1 being not used and 0 being used (meaning data
 * is valid and should be considered) 
 * - centerX - the x cordinate of the center
 * - centerY - the y cordinate of the center 
 * - endX - the x cordinate of the bottom right corner of the bounding box 
 * - endY - the y cordinate of the bottom right corner of the counding box 
 * - area - the area of the bounding box
 * - confidence - the confidence level of the neural network that the object is indeed what it is tagged as
 */

public class EVSNetworkTables extends SubsystemBase {
  // Put methods for controlling this subsystem
  // here. Call these from Commands.
  //String inUse = "inUse";
  //String values = "values";

  NetworkTableInstance n = NetworkTableInstance.getDefault();
  //NetworkTable evs;

  /* // Create and assign Power_Cell tables
  NetworkTable Power_Cell0;
  NetworkTable Power_Cell1;
  NetworkTable Power_Cell2;
  NetworkTable Power_Cell3;
  NetworkTable Power_Cell4;
  NetworkTable Power_Cell5;
  NetworkTable Power_Cell6;
  NetworkTable Power_Cell7;
  NetworkTable Power_Cell8;
  NetworkTable Power_Cell9;
  */

  @Override
  public void periodic() {
    // Set the default command for a subsystem here.
    // setDefaultCommand(new MySpecialCommand());

  }

  /*public void getVisionNetworkTable() {
  
      n = NetworkTableInstance.getDefault();
      evs = n.getTable("EVS");
  
      Power_Cell0 = evs.getSubTable("Power_Cell0");
      Power_Cell1 = evs.getSubTable("Power_Cell1");
      Power_Cell2 = evs.getSubTable("Power_Cell2");
      Power_Cell3 = evs.getSubTable("Power_Cell3");
      Power_Cell4 = evs.getSubTable("Power_Cell4");
      Power_Cell5 = evs.getSubTable("Power_Cell5");
      Power_Cell6 = evs.getSubTable("Power_Cell6");
      Power_Cell7 = evs.getSubTable("Power_Cell7");
      Power_Cell8 = evs.getSubTable("Power_Cell8");
      Power_Cell9 = evs.getSubTable("Power_Cell9");
  
      Goal0 = evs.getSubTable("Goal0");
  }*/

  public NetworkTable getVisionTable() {

    n = NetworkTableInstance.getDefault();
    NetworkTable evs = n.getTable("EVS");
    return evs;

  }

  public ArrayList<ArrayList<Double>> getPowerCellArray() {

    ArrayList<ArrayList<Double>> visionArray = new ArrayList<ArrayList<Double>>(10);
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());

    //if (getVisionTable().getEntry("checked").getBoolean(false)) {

    //Start if statement

    if (getVisionTable().getSubTable("Power_Cell0").getEntry("inUse").getBoolean(true)) {

      System.out.println("GetEntryAsString:");
      System.out.println(getVisionTable().getSubTable("Power_Cell0").getEntry("inUse").getType());

      double powerCellArray[] = getVisionTable().getSubTable("Power_Cell0").getEntry("values")
          .getDoubleArray(new double[7]);

      ArrayList<Double> vals = new ArrayList<Double>();

      System.out.println("I got past the if statement!!!\n\n\n");
      System.out.println(powerCellArray);
      for (int i = 0; i < 7; i++) {

        vals.add(powerCellArray[i]);

      }

      visionArray.set(0, vals);

      if (getVisionTable().getSubTable("Power_Cell1").getEntry("inUse").getBoolean(true)) {

        powerCellArray = getVisionTable().getSubTable("Power_Cell1").getEntry("values").getDoubleArray(new double[7]);

        vals = new ArrayList<Double>();
        for (int i = 0; i < 7; i++) {

          vals.add(powerCellArray[i]);

        }

        visionArray.set(1, vals);

        if (getVisionTable().getSubTable("Power_Cell2").getEntry("inUse").getBoolean(true)) {

          powerCellArray = getVisionTable().getSubTable("Power_Cell2").getEntry("values").getDoubleArray(new double[7]);

          vals = new ArrayList<Double>();
          for (int i = 0; i < 7; i++) {

            vals.add(powerCellArray[i]);

          }

          visionArray.set(2, vals);

          if (getVisionTable().getSubTable("Power_Cell3").getEntry("inUse").getBoolean(true)) {

            powerCellArray = getVisionTable().getSubTable("Power_Cell3").getEntry("values")
                .getDoubleArray(new double[7]);

            vals = new ArrayList<Double>();
            for (int i = 0; i < 7; i++) {

              vals.add(powerCellArray[i]);

            }

            visionArray.set(3, vals);

            if (getVisionTable().getSubTable("Power_Cell4").getEntry("inUse").getBoolean(true)) {

              powerCellArray = getVisionTable().getSubTable("Power_Cell4").getEntry("values")
                  .getDoubleArray(new double[7]);

              vals = new ArrayList<Double>();
              for (int i = 0; i < 7; i++) {

                vals.add(powerCellArray[i]);

              }

              visionArray.set(4, vals);

              if (getVisionTable().getSubTable("Power_Cell5").getEntry("inUse").getBoolean(true)) {

                powerCellArray = getVisionTable().getSubTable("Power_Cell5").getEntry("values")
                    .getDoubleArray(new double[7]);

                vals = new ArrayList<Double>();
                for (int i = 0; i < 7; i++) {

                  vals.add(powerCellArray[i]);

                }

                visionArray.set(5, vals);

                if (getVisionTable().getSubTable("Power_Cell6").getEntry("inUse").getBoolean(true)) {

                  powerCellArray = getVisionTable().getSubTable("Power_Cell6").getEntry("values")
                      .getDoubleArray(new double[7]);

                  vals = new ArrayList<Double>();
                  for (int i = 0; i < 7; i++) {

                    vals.add(powerCellArray[i]);

                  }

                  visionArray.set(6, vals);

                  if (getVisionTable().getSubTable("Power_Cell7").getEntry("inUse").getBoolean(true)) {

                    powerCellArray = getVisionTable().getSubTable("Power_Cell7").getEntry("values")
                        .getDoubleArray(new double[7]);

                    vals = new ArrayList<Double>();
                    for (int i = 0; i < 7; i++) {

                      vals.add(powerCellArray[i]);

                    }

                    visionArray.set(7, vals);

                    if (getVisionTable().getSubTable("Power_Cell8").getEntry("inUse").getBoolean(true)) {

                      powerCellArray = getVisionTable().getSubTable("Power_Cell8").getEntry("values")
                          .getDoubleArray(new double[7]);

                      vals = new ArrayList<Double>();
                      for (int i = 0; i < 7; i++) {

                        vals.add(powerCellArray[i]);

                      }

                      visionArray.set(8, vals);

                      if (getVisionTable().getSubTable("Power_Cell9").getEntry("inUse").getBoolean(true)) {

                        powerCellArray = getVisionTable().getSubTable("Power_Cell9").getEntry("values")
                            .getDoubleArray(new double[7]);

                        vals = new ArrayList<Double>();
                        for (int i = 0; i < 7; i++) {

                          vals.add(powerCellArray[i]);

                        }

                        visionArray.set(9, vals);

                      }
                    } //End 9
                  } //End 8
                } //End 7
              } //End 6
            } //End 5
          } //End 4
        } //End 3
      } //End 2
    } //End 1
      //End if statement

    //}

    return visionArray;

  }

  public ArrayList<ArrayList<Double>> getGoalArray() {

    ArrayList<ArrayList<Double>> visionArray = new ArrayList<ArrayList<Double>>(10); //We gotta initialize something in each spot
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    visionArray.add(new ArrayList<Double>());
    //if (getVisionTable().getEntry("checked").getBoolean(false)) {

    //Start if statement

    if (getVisionTable().getSubTable("Goal0").getEntry("inUse").getBoolean(false)) {

      double goalArray[] = getVisionTable().getSubTable("Goal0").getEntry("values").getDoubleArray(new double[7]);

      ArrayList<Double> vals = new ArrayList<Double>();

      for (int i = 0; i < 7; i++) {

        vals.add(goalArray[i]);

      }

      visionArray.set(0, vals);
    }
    // }
    return visionArray;
  }
  /*public double[][] getAllObjects() {
  
  double[][] allValues = new double[12][7];
  return allValues;
  
  }*/

  /*
    public double[][] getPower_Cell() {
  getVisionNetworkTable();
  
  Power_Cell_inUse0 = Power_Cell0.getEntry(inUse);
  Power_Cell_inUse1 = Power_Cell1.getEntry(inUse);
  Power_Cell_inUse2 = Power_Cell2.getEntry(inUse);
  Power_Cell_inUse3 = Power_Cell3.getEntry(inUse);
  Power_Cell_inUse4 = Power_Cell4.getEntry(inUse);
  Power_Cell_inUse5 = Power_Cell5.getEntry(inUse);
  Power_Cell_inUse6 = Power_Cell6.getEntry(inUse);
  Power_Cell_inUse7 = Power_Cell7.getEntry(inUse);
  Power_Cell_inUse8 = Power_Cell8.getEntry(inUse);
  Power_Cell_inUse9 = Power_Cell9.getEntry(inUse);
  
  double[][] Power_CellValues = new double[10][7]; 
  
  // if inUse is true, store values and check next table
  
  if (Power_Cell_inUse0.getBoolean(false) == true) {
    double Power_Cell0_values_array[] = Power_Cell_values0.getDoubleArray(new double[6]);
  
    for (int i = 0; i < 7; i++) {
      if (i == 0) {
        Power_CellValues[0][i] = 0;
      } else {
        Power_CellValues[0][i] = Power_Cell0_values_array[i - 1];
      }
  
    }
  
      if (Power_Cell_inUse1.getBoolean(false) == true) {
        double Power_Cell1_values_array[] = Power_Cell_values1.getDoubleArray(new double[6]);
  
        for (int i = 0; i < 7; i++) {
          if (i == 0) {
            Power_CellValues[1][i] = 0;
          } else {
            Power_CellValues[1][i] = Power_Cell1_values_array[i - 1];
          }
  
        }
  
          if (Power_Cell_inUse2.getBoolean(false) == true) {
            double Power_Cell2_values_array[] = Power_Cell_values2.getDoubleArray(new double[6]);
  
            for (int i = 0; i < 7; i++) {
              if (i == 0) {
                Power_CellValues[2][i] = 0;
              } else {
                Power_CellValues[2][i] = Power_Cell2_values_array[i - 1];
              }
      
            }
  
              if (Power_Cell_inUse3.getBoolean(false) == true) {
                double Power_Cell3_values_array[] = Power_Cell_values3.getDoubleArray(new double[6]);
  
                for (int i = 0; i < 7; i++) {
                  if (i == 0) {
                    Power_CellValues[3][i] = 0;
                  } else {
                    Power_CellValues[3][i] = Power_Cell3_values_array[i - 1];
                  }
          
                }
  
                  if (Power_Cell_inUse4.getBoolean(false) == true) {
                    double Power_Cell4_values_array[] = Power_Cell_values4.getDoubleArray(new double[6]);
  
                    for (int i = 0; i < 7; i++) {
                      if (i == 0) {
                        Power_CellValues[4][i] = 0;
                      } else {
                        Power_CellValues[4][i] = Power_Cell4_values_array[i - 1];
                      }
              
                    }
  
                      if (Power_Cell_inUse5.getBoolean(false) == true) {
                        double Power_Cell5_values_array[] = Power_Cell_values5.getDoubleArray(new double[6]);
  
                        for (int i = 0; i < 7; i++) {
                          if (i == 0) {
                            Power_CellValues[5][i] = 0;
                          } else {
                            Power_CellValues[5][i] = Power_Cell5_values_array[i - 1];
                          }
                  
                        }
  
                          if (Power_Cell_inUse6.getBoolean(false) == true) {
                            double Power_Cell6_values_array[] = Power_Cell_values6.getDoubleArray(new double[6]);
  
                            for (int i = 0; i < 7; i++) {
                              if (i == 0) {
                                Power_CellValues[6][i] = 0;
                              } else {
                                Power_CellValues[6][i] = Power_Cell6_values_array[i - 1];
                              }
                      
                            }
  
                              if (Power_Cell_inUse7.getBoolean(false) == true) {
                                double Power_Cell7_values_array[] = Power_Cell_values7.getDoubleArray(new double[6]);
  
                                for (int i = 0; i < 7; i++) {
                                  if (i == 0) {
                                    Power_CellValues[7][i] = 0;
                                  } else {
                                    Power_CellValues[7][i] = Power_Cell7_values_array[i - 1];
                                  }
                          
                                }
  
                                  if (Power_Cell_inUse8.getBoolean(false) == true) {
                                    double Power_Cell8_values_array[] = Power_Cell_values8.getDoubleArray(new double[6]);
  
                                    for (int i = 0; i < 7; i++) {
                                      if (i == 0) {
                                        Power_CellValues[8][i] = 0;
                                      } else {
                                        Power_CellValues[8][i] = Power_Cell8_values_array[i - 1];
                                      }
                              
                                    }
  
                                      if (Power_Cell_inUse9.getBoolean(false) == true) {
                                        double Power_Cell9_values_array[] = Power_Cell_values9.getDoubleArray(new double[6]);
  
                                        for (int i = 0; i < 7; i++) {
                                          if (i == 0) {
                                            Power_CellValues[9][i] = 0;
                                          } else {
                                            Power_CellValues[9][i] = Power_Cell9_values_array[i - 1];
                                          }
                                  
                                        }
  
                                      } else {
  
                                        Power_CellValues[0][0] = -1;
                                        Power_CellValues[1][0] = -1;
                                        Power_CellValues[2][0] = -1;
                                        Power_CellValues[3][0] = -1;
                                        Power_CellValues[4][0] = -1;
                                        Power_CellValues[5][0] = -1;
                                        Power_CellValues[6][0] = -1;
                                        Power_CellValues[7][0] = -1;
                                        Power_CellValues[8][0] = -1;
                                        Power_CellValues[9][0] = -1;
  
                                      }
  
                                  } else {
  
                                    Power_CellValues[0][0] = -1;
                                    Power_CellValues[1][0] = -1;
                                    Power_CellValues[2][0] = -1;
                                    Power_CellValues[3][0] = -1;
                                    Power_CellValues[4][0] = -1;
                                    Power_CellValues[5][0] = -1;
                                    Power_CellValues[6][0] = -1;
                                    Power_CellValues[7][0] = -1;
                                    Power_CellValues[8][0] = -1;
  
                                  }
  
                              } else {
  
                                Power_CellValues[0][0] = -1;
                                Power_CellValues[1][0] = -1;
                                Power_CellValues[2][0] = -1;
                                Power_CellValues[3][0] = -1;
                                Power_CellValues[4][0] = -1;
                                Power_CellValues[5][0] = -1;
                                Power_CellValues[6][0] = -1;
                                Power_CellValues[7][0] = -1;
  
                              }
  
                          } else {
  
                            Power_CellValues[0][0] = -1;
                            Power_CellValues[1][0] = -1;
                            Power_CellValues[2][0] = -1;
                            Power_CellValues[3][0] = -1;
                            Power_CellValues[4][0] = -1;
                            Power_CellValues[5][0] = -1;
                            Power_CellValues[6][0] = -1;
  
                          }
  
                      } else {
  
                        Power_CellValues[0][0] = -1;
                        Power_CellValues[1][0] = -1;
                        Power_CellValues[2][0] = -1;
                        Power_CellValues[3][0] = -1;
                        Power_CellValues[4][0] = -1;
                        Power_CellValues[5][0] = -1;
  
                      }
  
                  } else {
  
                    Power_CellValues[0][0] = -1;
                    Power_CellValues[1][0] = -1;
                    Power_CellValues[2][0] = -1;
                    Power_CellValues[3][0] = -1;
                    Power_CellValues[4][0] = -1;
  
                  }
  
              } else {
  
                Power_CellValues[0][0] = -1;
                Power_CellValues[1][0] = -1;
                Power_CellValues[2][0] = -1;
                Power_CellValues[3][0] = -1;
  
              }
  
          } else {
  
            Power_CellValues[0][0] = -1;
            Power_CellValues[1][0] = -1;
            Power_CellValues[2][0] = -1;
  
          }
  
      } else {
  
        Power_CellValues[0][0] = -1;
        Power_CellValues[1][0] = -1;
  
      }
  
  } else {
  
    Power_CellValues[0][0] = -1;
  
  }
  
    public double[][] getGoal() {
  getVisionNetworkTable();
  
  Goal_inUse0 = Goal0.getEntry(inUse);
  
  double[][] GoalValues = new double[1][7]; 
  
  // if inUse is true, store values and check next table
  
  if (Goal_inUse0.getBoolean(false) == true) {
    double Goal0_values_array[] = Goal_values0.getDoubleArray(new double[6]);
  
    for (int i = 0; i < 7; i++) {
      if (i == 0) {
        GoalValues[0][i] = 0;
      } else {
        GoalValues[0][i] = Goal0_values_array[i - 1];
      }
  
    }
  
  } else {
  
    GoalValues[0][0] = -1;
  
  }*/
}
