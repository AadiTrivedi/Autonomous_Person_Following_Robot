#!/usr/bin/env python3
#Node 6 Motor Driver

#Subscribes to /cmd_vel

#Outputs: GPIO PWM signals to L298N motor driver board

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

try:
    import RPi.GPIO as GPIO

    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
        GPIO_AVAILABLE = False


class MotorDriverNode(Node):
    def __init__(self):
        super().__init__("motor_driver")

        self.declare_parameter("left_enable_pin", 18)   
        self.declare_parameter("left_forward_pin", 23)   
        self.declare_parameter("left_reverse_pin", 24)   
        self.declare_parameter("right_enable_pin", 12)   
        self.declare_parameter("right_forward_pin", 25)  
        self.declare_parameter("right_reverse_pin", 22)  
        self.declare_parameter("wheel_base", 0.17)
        self.declare_parameter("max_motor_speed", 0.3)
        self.declare_parameter("pwm_frequency", 1000)
        self.left_en = self.get_parameter("left_enable_pin").value
        self.left_fwd = self.get_parameter("left_forward_pin").value
        self.left_rev = self.get_parameter("left_reverse_pin").value
        self.right_en = self.get_parameter("right_enable_pin").value
        self.right_fwd = self.get_parameter("right_forward_pin").value
        self.right_rev = self.get_parameter("right_reverse_pin").value
        self.wheel_base = self.get_parameter("wheel_base").value
        self.max_speed = self.get_parameter("max_motor_speed").value
        self.pwm_freq = self.get_parameter("pwm_frequency").value

        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            for pin in [
                self.left_en, self.left_fwd, self.left_rev,
                self.right_en, self.right_fwd, self.right_rev,
            ]:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW) 

            self.left_pwm = GPIO.PWM(self.left_en, self.pwm_freq)
            self.right_pwm = GPIO.PWM(self.right_en, self.pwm_freq)

            self.left_pwm.start(0)
            self.right_pwm.start(0)

            self.get_logger().info("GPIO initialized, motors ready")
        else:
            self.left_pwm = None
            self.right_pwm = None
            self.get_logger().warn("RPi.GPIO not available commands will be logged")


        self.cmd_sub = self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)

# stop if no input /cmd_vel
        self.watchdog_timer = self.create_timer(0.5, self.watchdog)
        self.last_cmd_time = self.get_clock().now()

        self.get_logger().info("Motor driver node ready")

    def on_cmd_vel(self, msg: Twist):
        self.last_cmd_time = self.get_clock().now()

        linear = msg.linear.x
        angular = msg.angular.z

        left_vel = linear - (angular * self.wheel_base / 2.0)
        right_vel = linear + (angular * self.wheel_base / 2.0)

        left_duty = self.velocity_to_duty(left_vel)
        right_duty = self.velocity_to_duty(right_vel)
        left_forward = left_vel >= 0
        right_forward = right_vel >= 0
        self.set_motor(
            self.left_pwm, self.left_fwd, self.left_rev,
            abs(left_duty), left_forward
        )
        self.set_motor(
            self.right_pwm, self.right_fwd, self.right_rev,
            abs(right_duty), right_forward
        )

        self.get_logger().debug(
            f"L: {left_vel:.2f}m/s ({left_duty:.0f}%) "
            f"R: {right_vel:.2f}m/s ({right_duty:.0f}%)"
        )

    def velocity_to_duty(self, velocity):
        duty = (velocity / self.max_speed) * 100.0
        return max(-100.0, min(100.0, duty))

    def set_motor(self, pwm, fwd_pin, rev_pin, duty, forward):
        if not GPIO_AVAILABLE:
            direction = "FWD" if forward else "REV"
            self.get_logger().info(
                f"[SIM] Motor: {direction} @ {duty:.0f}%"
            )
            return

        if forward:
            GPIO.output(fwd_pin, GPIO.HIGH)
            GPIO.output(rev_pin, GPIO.LOW)
        else:
            GPIO.output(fwd_pin, GPIO.LOW)
            GPIO.output(rev_pin, GPIO.HIGH)

        pwm.ChangeDutyCycle(duty)

    def watchdog(self):
        elapsed = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9

        if elapsed > 0.5:
            self.stop_motors()

    def stop_motors(self):
# Stop both motors
        if GPIO_AVAILABLE:
            self.left_pwm.ChangeDutyCycle(0)
            self.right_pwm.ChangeDutyCycle(0)
            GPIO.output(self.left_fwd, GPIO.LOW)
            GPIO.output(self.left_rev, GPIO.LOW)
            GPIO.output(self.right_fwd, GPIO.LOW)
            GPIO.output(self.right_rev, GPIO.LOW)

    def destroy_node(self):
        self.stop_motors()
        if GPIO_AVAILABLE:
            self.left_pwm.stop()
            self.right_pwm.stop()
            GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()