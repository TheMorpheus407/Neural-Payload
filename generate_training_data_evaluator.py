import json
import evaluator
import time
import requests
import util
import os

mydir = "Sites/"

def get_target_position(url, base_dict, beginn_string, end_string):
    resp = requests.post(url, base_dict)
    site = resp.text.replace("\t", "").replace("\r\n", "").replace("\n", "")
    beginn = site.find(beginn_string)
    end = site.find(end_string) + len(end_string)
    return str(beginn) + "-" + str(end)

def is_duplicate(base_inject, line):
    for i in all_data:
        if i["payload"] == util.payloaddict_to_string(base_inject, [line]):
            return True
    return False

def all_payloads(all_data):
    with open("traindata.txt", "r") as f:
        gen_data(f, all_data, False)

def xss(all_data):
    with open("negativepayloads", "r") as f:
        gen_data(f, all_data, True)

def gen_data(f, all_data, xss):
    for line in f.readlines():
        line = line.replace("\r\n", "").replace("\n", "")
        if is_duplicate(base_inject, line):
            continue
        res = evaler(line, target="post", xss=xss)
        time.sleep(0.01)
        file_path = mydir + url.split("/")[-1] + ".raw"
        raw = requests.get(url)
        if not os.path.isfile(file_path):
            open(file_path, "w").write(util.prepare_headers(raw.headers) + " \n " + raw.text)

        payload_dict = util.payloaddict_to_string(base_inject, [line])

        i = 0
        attack_path = mydir + url.split("/")[-1] + "_" + str(i) + ".raw"
        while os.path.isfile(attack_path):
            i = i + 1
            attack_path = mydir + url.split("/")[-1] + "_" + str(i) + ".raw"

        attk = requests.post(url, data=payload_dict)
        open(attack_path, "w").write(util.prepare_headers(attk.headers) + " \n " + attk.text)

        if res != True and res != False:
            #POSITIVE-Data
            all_data.append({"file": file_path, "attacked_file":attack_path, "payload": payload_dict, "target": target, "method": "post"})
        else:
            #NEGATIVE-Data
            all_data.append({"file": file_path, "attacked_file":attack_path, "payload": payload_dict, "target": "-", "method": "post"})



if __name__ == "__main__":
    generator = json.load(open("generator_information.json"))
    with open("website_attacks.txt", "r") as f:
        all_data = json.load(f)
    for i in generator:
        url = i["url"]
        if not "Masterarbeit" in url:
            continue
            input("Switch to docker container and press enter.")
        base_inject = i["base_injection_dict"]
        target = get_target_position(url, base_inject, i["start_string"], i["end_string"])
        evaler = evaluator.BasicEvaluator(url, base_inject)
        if i["xss"] == "True":
            xss(all_data)
        else:
            all_payloads(all_data)
    f = open("website_attacks.txt", "w")
    json.dump(all_data, f, indent=2)