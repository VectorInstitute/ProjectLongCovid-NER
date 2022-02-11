from m3inference import M3Twitter
import copy

class DemoInferer():
    def __init__(self):
        self.m3twitter = M3Twitter(skip_logging=True)

    def get_inferer_inputs(self, users):
        users = copy.deepcopy(users)
        for i in range(len(users)):
            users[i]["lang"] = "en"
            users[i]["id_str"] = str(users[i]["id"])
            users[i]["screen_name"] = str(users[i]["username"])

        inference_inputs = [self.m3twitter.transform_jsonl_object(u,
                                                img_path_key="profile_image_url",
                                                lang_key="lang") for u in users]
        return inference_inputs


    def infer(self, inputs):
        return self.m3twitter.infer(inputs, batch_size=256)