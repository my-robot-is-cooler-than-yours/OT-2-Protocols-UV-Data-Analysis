import paramiko
import subprocess
import time
from tkinter import Tk
from tkinter import filedialog


def run_subprocess(protocol_path):
    """
    Uses the subprocess module to transfer a protocol file to the OT-2 using legacy SCP (Secure Copy Protocol).
    This function is useful for uploading files from your local machine to the OT-2 before executing the protocol.

    :return: None
    """
    # Define the SCP command with the -O flag
    scp_command = [
        "scp",
        "-i", r"C:\Users\Lachlan Alexander\ot2_ssh_key",  # Path to your SSH key
        "-O",  # Force the legacy SCP protocol
        rf"{protocol_path}",
        # Local file path
        "root@169.254.80.171:/data/user_storage/prd_protocols"  # Destination on OT-2
    ]

    # Run the command using subprocess
    try:
        result = subprocess.run(scp_command, check=True, text=True, capture_output=True)
        print("File transferred successfully!")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print(f"Error output: {e.stderr}")


def run_execute(protocol_name):
    """
    Establish an SSH connection to the Opentrons OT-2 and execute a protocol via SSH.
    This function uses Paramiko to communicate with the OT-2, executes the given protocol, and then
    processes the output to check for the "Protocol Finished" message.

    :return: None
    """
    try:
        # Replace these with your own details
        hostname = "169.254.80.171"  # Replace with your OT-2's IP address
        username = "root"  # OT-2 default username is 'root'
        key_path = r"C:\Users\Lachlan Alexander\ot2_ssh_key"  # Path to your SSH private key
        # protocol_path = f"'/data/user_storage/prd_protocols/{protocol_name}.py'"  # Path to your protocol on the OT-2
        protocol_path = f"'/data/user_storage/prd_protocols/{protocol_name}'"  # Path to your protocol on the OT-2

        # If using a passphrase with your SSH key
        key_passphrase = ""  # Replace with your SSH key passphrase or None if no passphrase

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load the private key
        # private_key = paramiko.RSAKey.from_private_key_file(key_path, password=key_passphrase)

        ssh.connect(hostname, username=username, key_filename=key_path)

        stdin, stdout, stderr = ssh.exec_command(f'sh -l -c "opentrons_execute {protocol_path}"')

        # List to store the entire output
        full_output = []

        # Real-time output processing versus all-in-one at the end
        while True:
            line = stdout.readline()  # Read each line as it's received
            if not line:  # Break the loop when there's no more output
                break
            print(line, end='')  # Print the output line by line without extra newlines

            # Store the line in the full output list
            full_output.append(line)

            # Check for "Protocol Finished" in the real-time output
            if "Protocol Finished" in line:
                print("Protocol end detected")
                break

            # Small delay to avoid overloading the loop
            time.sleep(0.1)

        # Read the output from stdout
        output = ''.join(stdout.readlines())

        print(full_output)  # Optionally still print the entire output for debugging or completion detection

        # Check for the phrase "Protocol Finished"
        if ' Protocol Finished\n' in full_output:
            print("Protocol end detected")

        else:
            print("Protocol end not detected")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the SSH connection
        stdin.close()
        ssh.close()


def run_simulation(protocol_name):
    """
    Establish an SSH connection to the Opentrons OT-2 and execute a protocol via SSH.
    This function uses Paramiko to communicate with the OT-2, executes the given protocol, and then
    processes the output to check for the "Protocol Finished" message.

    :return: None
    """
    try:
        # Replace these with your own details
        hostname = "169.254.80.171"  # Replace with your OT-2's IP address
        username = "root"  # OT-2 default username is 'root'
        key_path = r"C:\Users\Lachlan Alexander\ot2_ssh_key"  # Path to your SSH private key
        # protocol_path = f"'/data/user_storage/prd_protocols/{protocol_name}.py'"  # Path to your protocol on the OT-2
        protocol_path = f"'/data/user_storage/prd_protocols/{protocol_name}'"  # Path to your protocol on the OT-2

        # If using a passphrase with your SSH key
        key_passphrase = ""  # Replace with your SSH key passphrase or None if no passphrase

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load the private key
        # private_key = paramiko.RSAKey.from_private_key_file(key_path, password=key_passphrase)

        ssh.connect(hostname, username=username, key_filename=key_path)

        stdin, stdout, stderr = ssh.exec_command(f'sh -l -c "opentrons_simulate {protocol_path}"')

        # List to store the entire output
        full_output = []

        # Real-time output processing versus all-in-one at the end
        while True:
            line = stdout.readline()  # Read each line as it's received
            if not line:  # Break the loop when there's no more output
                break
            print(line, end='')  # Print the output line by line without extra newlines

            # Store the line in the full output list
            full_output.append(line)

            # Check for "Protocol Finished" in the real-time output
            if "Protocol Finished" in line:
                print("Protocol end detected")
                break

            # Small delay to avoid overloading the loop
            time.sleep(0.1)

        # Read the output from stdout
        output = ''.join(stdout.readlines())

        print(full_output)  # Optionally still print the entire output for debugging or completion detection

        # Check for the phrase "Protocol Finished"
        if ' Protocol Finished\n' in full_output:
            print("Protocol end detected")

        else:
            print("Protocol end not detected")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the SSH connection
        stdin.close()
        ssh.close()


def get_file_path():
    """
    Prompt the user to select a file.

    :return: Full path of the selected file.
    """
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window

    while True:
        # Prompt the user to select a file
        file_name = filedialog.askopenfilename(title="Select a File")

        if not file_name:
            print("No file selected, please select a file.")
        else:
            break

    root.quit()  # Close the Tkinter root window
    return file_name


if __name__ == "__main__":
    # file_path = get_file_path()
    # file_name = file_path.split("/")[-1]
    # print(file_name)

    run_subprocess(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\LCST\23-Jan full plate + salt + HCl\32.5 C Predicted Mixture\Duplicated_Volumes.csv")
    run_subprocess(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Honours Python Main\OT-2 Protocols\DoE + Monomers Experiment\Mixtures Multivariable.py")

    run_simulation("Mixtures Multivariable.py")
