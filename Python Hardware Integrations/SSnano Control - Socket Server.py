import os
import time
import socket

current_directory = os.getcwd()


def timeit(func):
    """
    Decorator for measuring function's running time.
    """

    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        log_msg(f"Processing time of {func.__qualname__}(): {time.time() - start_time:.2f} seconds.")
        return result

    return measure_time


def log_msg(message: str):
    """
    Log a message with a timestamp.

    :param message: String, the message to be logged.
    :return: None
    """
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{current_time}] {message}")


def send_message(conn, message_type: str, message_data: str = ""):
    """
    Send a message to the client with a specified message type.

    The message is encoded as a string that combines the message type and optional message data, separated by a pipe ('|') character.
    The message is then sent to the server through the provided socket.

    :param conn: The socket object used to communicate with the server.
    :param message_type: The type of the message being sent (e.g., 'REQUEST', 'UPDATE').
    :param message_data: Optional additional data to include with the message (default is an empty string).
    :return: None
    """
    message = f"{message_type}|{message_data}"
    conn.sendall(message.encode())


def receive_message(conn):
    """
    Receive a message from the server.

    This function waits to receive a message from the server via the provided socket. The received message is split into
    its type and data components, using the pipe ('|') character as a delimiter.

    :param conn: The socket object used to receive the message.
    :return: A tuple containing the message type and message data.
    """
    data = conn.recv(1024).decode()
    return data.split("|", 1)


def example_workflow(conn):
    """
    Workflow for managing the interaction between the server and a client during
    a sample preparation and measurement process. This function is designed to be used within
    handle_client() and follows the sequence of tasks such as taking a background reading,
    preparing samples, running measurement protocols, and using machine learning for analysis.

    The workflow prompts the user for confirmations to proceed with each step,
    communicates with the plate reader, and processes CSV files received from the client.

    :param conn: A socket connection object used to communicate with the client.
    :return: None
    """

    # Define output path for data
    out_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\Automated Testing"
    verification = False

    while True:

        # Ask plate reader to take blank reading of plate
        # Wait for user confirmation to proceed

        log_msg("Plate background is required to be taken before proceeding")

        while True:
            user_input = input(">>> Has empty plate been prepared and ready to be inserted? (yes/no): \n>>> ")
            if user_input.lower() == "yes":
                break
            else:
                log_msg("Waiting for plate preparation...")

        log_msg("Requesting plate background from reader")
        send_message(conn, "PLATE_BACKGROUND", "Empty Plate Reading")

        log_msg("Awaiting message from client")
        msg_type, msg_data = receive_message(conn)

        if msg_type == "PLATE_BACKGROUND":
            log_msg("Plate background data received")

        # Save plate bg to variable
        plate_background_path = msg_data
        log_msg("Plate background path saved")

        ## Make robot prepare samples now ##
        log_msg("Please allow robot to prepare samples before proceeding")

        # Wait for user confirmation to proceed
        while True:
            user_input = input(">>> Has full plate been prepared and ready to be inserted? (yes/no): \n>>> ")
            if user_input.lower() == "yes":
                break
            else:
                log_msg("Waiting for plate preparation...")

        log_msg("Requesting to run measurement protocol")
        send_message(conn, "RUN_PROTOCOL", "Empty Plate Reading")

        log_msg("Awaiting message from client")
        msg_type, msg_data = receive_message(conn)

        if msg_type == "CSV_FILE":
            if verification is False:
                log_msg(f"Measurement complete")
                log_msg(f"Received CSV file with path: {msg_data}")

                data_path = msg_data
                log_msg("Data path saved")

                # Run machine learning screening
                log_msg("Doing ML stuff")

            elif verification is True:
                log_msg("!!!Verification Step!!!")
                log_msg(f"Measurement complete")
                log_msg(f"Received CSV file with path: {msg_data}")

                data_path = msg_data
                log_msg("Data path saved")

                # Run machine learning screening
                log_msg("Doing ML stuff")
                log_msg("Verification step complete")

        # Test certain conditions being met after analysis completed
        test = input(
            ">>> Condition met? \n>>> ")  # Imagine this is some condition being met from the ML quality parameters

        if test == "yes":
            pass
            verification = True
            # loop repeats to take set of verification samples

        else:
            verification = False  # this should break loop
            break  # just in case


@timeit
def handle_client(conn):
    """
    Handle communication with a client (64-bit script).

    This function manages the interaction between the BMG SPECTROstar Nano reader and the server. It listens for
    messages from the server and executes the appropriate actions, such as performing background readings,
    running protocols, and collecting sample data. Depending on the message type, it performs the necessary operations
    on the plate reader and sends back CSV data or other responses to the server.

    :param conn: A socket object used to communicate with the server.
    :raises Exception: If there is a failure in communicating with the client or server-side operations.
    :return: None
    """
    try:
        while True:

            choice = input(">>> Enter workflow number: \n1. Conc Model \n2. Shutdown \n>>> ")

            if choice == "1":
                example_workflow(conn)

            if choice == "2":
                send_message(conn, "SHUTDOWN")
                break

            else:
                log_msg("Invalid entry, please enter a number only.")

    except Exception as e:
        log_msg(f"Error handling client: {e}")


def server_main():
    """
    Main function to establish communication between the 64-bit server and the 32-bit client.

    This function initializes the connection to the client. It handles the full
    lifecycle of the client-server interaction and ensures proper connection termination.

    :raises Exception: If an error occurs during communication or setup.
    :return: None
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 65432))  # Start the server
            s.listen()
            log_msg("Waiting for connection from 32-bit script...")

            # while True:
            conn, addr = s.accept()
            with conn:
                log_msg(f"Connected by {addr}")

                handle_client(conn)

                log_msg("Shutting down.")
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()

    except Exception as e:
        log_msg(f"An error occurred: {e}")


if __name__ == "__main__":
    server_main()
