from .base_dataset import BaseDataset


class VQAMMEHRDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqa_mmehr_train"]
        elif split == "val":
            names = ["vqa_mmehr_val"]        
        elif split == "test":
            names = ["vqa_mmehr_test"]           
        
        if kwargs["exp_name"] != "task_finetune_vqa_mmehr":
            suffix = kwargs["exp_name"].split("task_finetune_vqa_mmehr_")[-1]
            if suffix == "ref":
                if split in ["val", "test"]:
                    names = [name + f"_{suffix}" for name in names]
            else:
                names = [name + f"_{suffix}" for name in names]
        del(kwargs["exp_name"])
        
        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions"
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        answers = self.table["answers"][index][question_index].as_py()
        labels = self.table["answer_labels"][index][question_index].as_py()
        scores = self.table["answer_scores"][index][question_index].as_py()
        answer_types = self.table["answer_type"][index][question_index].as_py()
        content_types = self.table["content_type"][index][question_index].as_py()

        return {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "answer_types": answer_types,
            "content_types": content_types,
            "qid": question_index,
            "iid": index,
        }
