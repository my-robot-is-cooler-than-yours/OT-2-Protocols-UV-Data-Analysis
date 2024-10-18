import win32com.client.gencache
import time


def log_msg(message: str):
    """
    Log a message with a timestamp.

    :param message: String, the message to be logged.
    :return: None
    """
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{current_time}] {message}")


class BmgCom:
    """
    A class to handle communication with the BMG SPECTROstar Nano plate reader using ActiveX.

    This class provides methods to control the plate reader, such as opening connections, running protocols,
    setting temperature, and inserting/ejecting plates.

    :param control_name: Optional name of the device to connect to. If provided, an attempt to open
                         a connection is made during initialization.
    """

    def __init__(self, control_name: str = None):
        """
        Initialize the BmgCom class and create the ActiveX COM object.

        :param control_name: Optional string specifying the control name for the reader. If provided,
                             an automatic connection is attempted.
        :raises: Exception if the ActiveX COM object instantiation or connection fails.
        """
        try:
            # Initialize ActiveX COM object
            self.com = win32com.client.gencache.EnsureDispatch("BMG_ActiveX.BMGRemoteControl")
            log_msg("COM object created successfully.")

        except Exception as e:
            log_msg(f"Instantiation failed: {e}")
            raise

        if control_name:
            self.open(control_name)

    def open(self, control_name: str):
        """
        Open a connection to the BMG reader.

        :param control_name: String specifying the name of the reader to connect to.
        :raises: Exception if the connection fails or returns an error status.
        """
        try:
            result_status = self.com.OpenConnectionV(control_name)
            if result_status:
                raise Exception(f"OpenConnection failed: {result_status}")
            log_msg(f"Connected to {control_name} successfully.")
        except Exception as e:
            log_msg(f"Failed to open connection: {e}")
            raise

    def version(self):
        """
        Retrieve the software version of the BMG ActiveX Interface.

        :return: String representing the software version.
        :raises: Exception if the version retrieval fails.
        """
        try:
            version = self.com.GetVersion()
            log_msg(f"Software version: {version}")
            return version
        except Exception as e:
            log_msg(f"Failed to get version: {e}")
            raise

    def status(self):
        """
        Get the current status of the plate reader e.g., 'Ready', 'Busy'.

        :return: String representing the current status of the reader.
        :raises: Exception if the status retrieval fails.
        """
        try:
            status = self.com.GetInfoV("Status")
            return status.strip() if isinstance(status, str) else 'unknown'
        except Exception as e:
            log_msg(f"Failed to get status: {e}")
            raise

    def plate_in(self):
        """
        Insert the plate holder into the reader.

        :return: None
        :raises: Exception if the plate insertion command fails.
        """
        try:
            self.exec(['PlateIn'])
            log_msg("Plate inserted into the reader.")
        except Exception as e:
            log_msg(f"Failed to insert plate: {e}")
            raise

    def plate_out(self):
        """
        Eject the plate holder from the reader.

        :return: None
        :raises: Exception if the plate insertion command fails.
        """
        try:
            self.exec(['PlateOut'])
            log_msg("Plate ejected from the reader.")
        except Exception as e:
            log_msg(f"Failed to eject plate: {e}")
            raise

    def set_temp(self, temp: str):
        """
        Activate the plate reader's incubator and set it to a target temperature.
        Note that this command does not wait for the heating plates to reach the target temperature before proceeding.

        :return: None
        :raises: Exception if the plate insertion command fails.
        """
        try:
            self.exec(['Temp', temp])
            log_msg(f"Temperature set to {temp}.")
        except Exception as e:
            log_msg(f"Failed to set temperature: {e}")
            raise

    def run_protocol(self,
                     name: str,
                     test_path: str = r'C:\Program Files (x86)\BMG\SPECTROstar Nano\User\Definit',
                     data_path: str = r'C:\Program Files (x86)\BMG\SPECTROstar Nano\User\Data'
                     ):
        """
        Run a test protocol from pre-defined protocols stored on the plate reader.
        test_path and data_path variables should remain unchanged
            as these are default directories from BMG software install.

        :return: None
        :raises: Exception if the plate insertion command fails.
        """
        try:
            # self.exec(['Run', name, test_path, data_path])
            self.com.ExecuteAndWait(['Run', name, test_path, data_path])
            log_msg(f"Protocol '{name}' completed successfully.")
        except Exception as e:
            log_msg(f"Failed to run protocol '{name}': {e}")
            raise

    def exec(self, cmd: list):
        """
        Eject the plate holder from the reader.

        :return: None
        :raises: Exception if the plate insertion command fails.
        """
        try:
            res = self.com.ExecuteAndWait(cmd)
            if res:
                raise Exception(f"Command {cmd} failed: {res}")
        except Exception as e:
            log_msg(f"Command execution failed: {e}")
            raise


def control_example():
    """
    An example of controlling the BMG SPECTROstar Nano control workflow.

    This function initializes the BmgCom class, connects to the SPECTROstar Nano reader, and executes a series
    of operations, including ejecting/inserting the plate carrier, setting temperature, and running a
    predefined protocol.

    It provides a user prompt to verify that the plate has been inserted, ensuring the process flow can proceed.

    :raises: Exception if any step in the control sequence fails.
    :return: None
    """
    try:
        bmg = BmgCom("SPECTROstar Nano")

        version = bmg.version()
        log_msg(f"BMG ActiveX Interface Software version: {version}")

        log_msg(f"Instrument Status: {bmg.status()}")

        ver = ""
        while ver.lower() != "yes":
            log_msg('Ejecting Plate Carrier')
            bmg.plate_out()
            # If plate is already out, this step is skipped

            log_msg(f"Instrument Status: {bmg.com.GetInfoV('Status')}")

            ver = input(">>> Has plate been inserted? (yes/no): \n>>> ")

        log_msg('Input verified, proceeding')

        log_msg('Inserting Plate Carrier')
        bmg.plate_in()
        log_msg(f"Instrument Status: {bmg.com.GetInfoV('Status')}")

        target_temp = '25.0'
        log_msg(f"Setting Temperature to {target_temp}")
        bmg.com.ExecuteAndWait(['Temp', target_temp])

        protocol_name = 'Empty Plate Reading'
        test_runs_path = r'C:\Program Files (x86)\BMG\SPECTROstar Nano\User\Definit'
        data_output_path = r'C:\Program Files (x86)\BMG\SPECTROstar Nano\User\Data'

        log_msg(f"Running protocol with name {protocol_name}")
        bmg.com.ExecuteAndWait(['Run', protocol_name, test_runs_path, data_output_path])
        log_msg(f"Protocol {protocol_name} completed")
        log_msg(f"Instrument Status: {bmg.com.GetInfoV('Status')}")

    except Exception as e:
        log_msg(e)


if __name__ == '__main__':
    # Run example workflow
    control_example()
