import paramiko
import subprocess


def run_ssh_command_deprecated():
    """
    Establishes an SSH connection to the Opentrons OT-2 and executes a protocol on the robot.
    This function uses Paramiko for SSH connection and remote execution, allowing you to run
    a protocol located on the OT-2.

    This is a deprecated version of the SSH command function, using older techniques.

    :return: None
    """
    # Replace these with your own details
    hostname = ""  # Replace with your OT-2's IP address
    username = ""  # OT-2 default username is 'root'
    key_path = r""  # Path to your SSH private key
    protocol_path = ""  # Path to your protocol on the OT-2

    # If using a passphrase with your SSH key
    key_passphrase = ""  # Replace with your SSH key passphrase or None if no passphrase

    try:
        # Create an SSH client
        client = paramiko.SSHClient()

        # Auto add the OT-2's host key
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load the private key
        private_key = paramiko.RSAKey.from_private_key_file(key_path, password=key_passphrase)

        # Connect to the OT-2
        client.connect(hostname, username=username, pkey=private_key)

        # Define the command you want to execute
        command = f"opentrons_execute {protocol_path}"

        # Run the command on the OT-2
        stdin, stdout, stderr = client.exec_command(command)

        # Read and print the command output and errors
        output = stdout.read().decode()
        errors = stderr.read().decode()

        if output:
            print(f"Output:\n{output}")
        if errors:
            print(f"Errors:\n{errors}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the SSH connection
        client.close()


def run_ssh_command():
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
        protocol_path = "'/var/lib/jupyter/notebooks/LA Test Protocol 24-Jul_diff.py'"  # Path to your protocol on the OT-2

        # If using a passphrase with your SSH key
        key_passphrase = ""  # Replace with your SSH key passphrase or None if no passphrase

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load the private key
        # private_key = paramiko.RSAKey.from_private_key_file(key_path, password=key_passphrase)

        ssh.connect(hostname, username=username, key_filename=key_path)

        stdin, stdout, stderr = ssh.exec_command(f'sh -l -c "opentrons_execute {protocol_path}"')

        # Read the output from stdout
        output = ''.join(stdout.readlines())
        print(type(output))
        print(output)  # Optional: print the output for debugging

        # Check for the phrase "Protocol Finished"
        if "Protocol Finished" in output:
            print("Hello!")

        else:
            print("Not found!")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the SSH connection
        stdin.close()
        ssh.close()


def run_subprocess():
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
        r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Honours Python Main\OT-2 Protocols\LA Test Protocol 24-Jul_diff.py",
        # Local file path
        "root@169.254.80.171:/var/lib/jupyter/notebooks/"  # Destination on OT-2
    ]

    # Run the command using subprocess
    try:
        result = subprocess.run(scp_command, check=True, text=True, capture_output=True)
        print("File transferred successfully!")
        print(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print(f"Error output: {e.stderr}")


if __name__ == "__main__":
    # First, transfer the protocol file to the OT-2 using SCP
    run_subprocess()
    # Then, execute the protocol via SSH
    run_ssh_command()
