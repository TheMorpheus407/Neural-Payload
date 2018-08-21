import util
import requests

class BasicEvaluator():
    def __init__(self, url, base_injection_dict, base_html_headers = None, base_html_text = None):
        full_resp = requests.post(url, base_injection_dict)
        assert full_resp.status_code == 200

        if base_html_headers == None:
            self.base_html_headers = full_resp.headers
        else:
            self.base_html_headers = base_html_headers
        if base_html_text == None:
            self.base_html_text = full_resp.text
        else:
            self.base_html_text = base_html_text
        self.base_html_text = self.base_html_text
        self.base_html_headers = self.base_html_headers
        if "date" in self.base_html_headers:
            self.base_html_headers.pop("date")
        self.base_injection_dict = base_injection_dict
        self.url = url

    """
    :returns False if website does not respond or wrong usage
    :returns True if output is the same and if XSS: if dangerous tags are added through the payload
    :returns the output if different
    """
    def __call__(self, *args, **kwargs):
        #args are POST-Parameters
        if kwargs["target"] == "post":
            to_inject = util.payloaddict_to_string(self.base_injection_dict, args)
            resp = requests.post(self.url, data=to_inject)
            print("SENDING REQUEST TO " + self.url + " WITH PARAMS " + str(to_inject))
            headers = resp.headers
            if "date" in headers:
                headers.pop("date")
            if resp.status_code != 200:
                return False
            if resp.text == self.base_html_text and headers == self.base_html_headers:
                return True
            else:
                if "xss" in kwargs:
                    return self.xss(resp, args)
                return resp
        return False

    def xss(self, resp, args):
        legit_scripts = self.base_html_text.count("<script>")
        attack_scripts = resp.text.count("<script>")
        legit_alerts = self.base_html_text.count("alert(")
        attack_alerts = resp.text.count("alert(")
        if attack_scripts > legit_scripts or attack_alerts > legit_alerts:
            return resp
        else:
            return True