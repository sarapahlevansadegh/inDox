import gc

import numpy as np
import torch.cuda
from bert_score import BERTScorer
from collections import defaultdict
from unieval import DialogEvaluator

Example = {
    "query": ["How did Cinderella reach her happy ending ?"],
    "answer": ['''Cinderella reached her happy ending with the help of a magical bird that granted her wishes.
    Despite being mistreated by her stepmother and stepsisters, Cinderella's kindness and hopefulness prevailed. With
    the bird's assistance, she was able to attend a royal festival where the prince fell in love with her. Through a
    series of events and magical interventions, including being dressed in beautiful attire by the bird, Cinderella
    captured the heart of the prince and eventually rode away with him, revealing her true identity and finding her
    happily ever after.'''],
    "context": [
        '''The provided documentation describes the classic fairy tale of Cinderella. In the story, Cinderella is
    mistreated by her step-sisters and step-mother, but with the help of a magical bird that grants her wishes,
    she is able to attend a royal festival where the prince falls in love with her. With the bird's help, Cinderella
    attends the festival in increasingly beautiful dresses and eventually rides away with the prince. At the end of
    the tale, the true nature of Cinderella is revealed with''',
        '''The documentation provided is a retelling of the classic fairy tale of Cinderella. It describes how
    Cinderella, a kind and beautiful young girl, is mistreated by her stepmother and stepsisters but remains hopeful
    and kind. With the help of a magical bird that fulfills her wishes, Cinderella is able to attend a royal festival
    where the prince falls in love with her. Despite her stepsisters' attempts to deceive the prince, the truth is
    revealed with the help of the magical bird, and''',
        '''The provided text seems to be an excerpt from the classic fairy tale of Cinderella. It tells the story of a
    young girl who faces mistreatment from her stepmother and stepsisters. Despite their cruelty, she remains kind
    and caring. Her father, unable to recognize Cinderella, who he believes to be an unknown maiden, tries to find
    her by cutting down a tree and later trapping her in a pigeon-house. Eventually, through a series of events,
    the prince discovers Cinderella's true identity and''',
        '''The documentation provided includes multiple excerpts detailing the classic fairy tale of Cinderella. The
    story revolves around a young girl who faces mistreatment from her stepmother and stepsisters but remains kind
    and hopeful. Despite enduring hardships and cruel treatment, Cinderella's character shines through. The narrative
    describes how Cinderella, with the help of magical elements like a little white bird and a hazel tree,
    is able to attend a royal festival and capture the heart of the prince. The tale also touches upon the challenges
    Cinder''',
        '''The provided text is an excerpt from the famous fairy tale \"Cinderella.\" It narrates the story of
        Cinderella, who is mistreated by her stepmother and stepsisters but is aided by a magical hazel tree and a
        little white bird. When her stepsisters are invited to a royal festival to find a bride for the prince,
        Cinderella is left behind. The bird helps her get dressed in beautiful attire, and she attends the festival
        where she captivates everyone, including the prince '''
    ]
}

Example2 = {
    "query": ["How did Cinderella reach her happy ending ?"],
    "answer": ['''Cinderella reaches her happy ending in the classic fairy tale through a series of events that showcase her kindness, perseverance, and eventual justice. Despite facing mistreatment from her stepmother and stepsisters, Cinderella remains pious and good-hearted. With the help of magical elements like a hazel tree, birds, and a golden slipper, she is able to attend a royal festival where she captures the prince's attention.

The stepsisters try to deceive the prince by mutilating their feet to fit into the golden slipper, but they are exposed by the vigilant birds. Ultimately, Cinderella's true identity is revealed when the slipper fits her perfectly. The prince recognizes her as the true bride he had danced with before, and despite objections from her stepfamily, he marries Cinderella.

The story concludes with Cinderella finding happiness and love with the prince, while her stepsisters face consequences for their deceitful actions, such as being punished with blindness. This classic fairy tale emphasizes the triumph of goodness and purity of heart over cruelty and jealousy, leading Cinderella to her happily ever after with her true love.'''],
    "context": [
        "The document provided is a retelling of the classic fairytale of Cinderella. It starts with the premise of a rich man's wife falling ill and advising her daughter to be good and pious. After the mother's death, the daughter faces mistreatment from her stepmother and stepsisters, who force her to do hard labor and treat her as a kitchen maid. The story progresses as Cinderella is denied attending the royal festival multiple times until a magical hazel tree helps her by granting her beautiful clothes and shoes for the event. She captures the attention of the prince at the festival but has to flee each night, leaving behind a golden slipper in her haste.\n\nThe prince searches for the mysterious maiden whose foot fits the slipper. The stepsisters try to deceive the prince by mutilating their feet to fit into the shoe, but the two white pigeons sitting on the hazel tree expose them by revealing the blood. Finally, Cinderella's true identity is revealed when she fits perfectly into the golden slipper. The prince recognizes her as the true bride he had danced with before and, despite the objections of her stepmother and stepsisters, marries Cinderella. The stepsisters are punished for their wickedness with blindness for the rest of their lives. The fairytale ends with Cinderella marrying the prince and living happily ever after.",
        "The provided text seems to be an excerpt from the fairy tale of Cinderella. In this adapted version of the story, Cinderella's step-sisters try to force their feet into the golden slipper to deceive the prince. They both mutilate their feet in attempts to fit into the slipper but fail. Finally, Cinderella herself fits into the slipper and reveals her true identity to the prince. The story culminates in Cinderella marrying the prince and being vindicated by magical white doves. The retelling captures the classic elements of the Cinderella story, including the wicked step-family, the magical transformation, and the eventual happy ending where Cinderella marries her true love.",
        'The documentation provided is a detailed version of the classic fairy tale of Cinderella. It begins with the wife of a rich man on her deathbed advising her daughter to be good and pious, as her mother will watch over her from heaven. After her mother\'s death, the girl was mistreated by her stepmother and stepsisters, who forced her to do menial tasks and gave her the nickname "Cinderella." Despite the hardships, Cinderella remains kind and pure-hearted.\n\nThe story continues with the stepmother and stepsisters taunting Cinderella and promising her that she can attend the royal festival if she can pick lentils from the ashes within a certain time frame. With the help of the birds in the garden, Cinderella successfully completes the task. However, her stepmother denies her the chance to attend the festival, setting more impossible tasks for her.\n\nAs Cinderella proves herself capable through the help of the birds multiple times, her stepmother continues to find excuses to prevent her from joining them. Eventually, the stepmother and stepsisters face punishment for their wickedness and falsehood when the birds peck out their eyes.\n\nThe story ends with Cinderella finding favor with the prince and achieving a happy ending while her stepfamily faces the consequences of their cruelty. The tale emphasizes the themes of kindness, perseverance, and eventual justice.',
        "The documentation provided is a detailed account of the story of Cinderella. It begins with the death of Cinderella's mother and her mistreatment by her stepmother and stepsisters. Despite the hardships she faces, Cinderella remains pious and good. With the help of a magical hazel tree and birds, Cinderella is able to attend a royal festival where the prince falls in love with her but she disappears each night. On the third day of the festival, the prince follows her home, trying to identify her using a golden slipper she leaves behind.\n\nThe stepsisters try to fit into the slipper by cutting off parts of their feet, but are exposed by the vigilant birds. Ultimately, Cinderella is identified as the true bride when the slipper fits her perfectly. She marries the prince, and her stepsisters are punished by the pigeons, who peck out their eyes for their wickedness.\n\nThis classic fairy tale highlights the themes of kindness, perseverance, and justice. It shows that goodness and purity of heart will ultimately triumph over cruelty and jealousy. The story concludes with Cinderella finding happiness and love with the prince, while her stepsisters face consequences for their deceitful actions.",
        'The documentation provided is a detailed retelling of the classic fairy tale "Cinderella." It begins with Cinderella\'s father buying gifts for her step-sisters and giving her a hazel branch. Cinderella plants the branch on her mother\'s grave, and a tree grows from it that grants her wishes. When a royal festival is announced, Cinderella\'s step-sisters prepare to attend, and she asks to go as well. With the help of the bird in the tree, Cinderella is transformed and goes to the festival in a beautiful dress. The prince only dances with her, but she flees each night and leaves behind a golden slipper. The prince finds the slipper and sets out to find its owner, leading to the famous happy ending of the story.']
}
Example3 = {
    'query': ['how cinderella reach her happy ending ?'],
    'answer': ['Persian Gulf is surrounded by middle east countries'],
    'context': ['Persian Gulf is surrounded by middle east countries']
}


def convert_to_json(output_list, src_list=None, ref_list=None, context_list=None, scores=None, doc_id=None,
                    system_id=None):
    """
        Convert the data into the json format.

        output_list: a list of model output
        src_list: source input for different NLG tasks. For example, source document for summarization
                  and dialogue history for dialogue response generation
        ref_list: human-annotated groundtruth
        context_list: the context needed to evaluate several specific dimension. For example,
                      additional factual information when evaluating engagingness and groundedness in dialogues
        scores: human scores for evaluating the model output. They can be used to calculate the correlation
                between evaluators and human judgements. The scores should be stored in a dictionary. For example,
                {'fluency': 2.0, 'coherence': 3.0} could be the human score for a sample.
        doc_id: the index of the input source. It can be used to calculate summary-level correlation for summarzation
        system_id: the index of the generation system. It can be used to calculate system-level correlation.
    """
    json_data = []
    for i in range(len(output_list)):
        cur = {}
        cur['system_output'] = output_list[i]
        if src_list is not None:
            cur['source'] = src_list[i]
        if ref_list is not None:
            cur['reference'] = ref_list[i]
        if context_list is not None:
            cur['context'] = context_list[i]
        if scores is not None:
            cur['scores'] = scores[i]
        if doc_id is not None:
            cur['doc_id'] = doc_id[i]
        if system_id is not None:
            cur['system_id'] = system_id[i]
        json_data.append(cur)
    return json_data


class BertEvaluator(BERTScorer):
    def scores(self, inputs, verbose=False, batch_size=64, return_hash=False):
        _, answer, context = inputs['query'], inputs['answer'], inputs['context']
        cands = [answer[0] for _ in range(len(context))]
        scores = defaultdict(list)
        K = len(context)
        for i in range(K):
            P, R, F1 = self.score([cands[i]], [context[i]], verbose=False)
            [scores[key].append(value) for key, value in [('P', P.numpy()), ('R', R.numpy()), ('F1', F1.numpy())]]
            del P, R, F1
        scores['K'].append(K)
        mP, mR, mF1 = np.array(scores['P']).mean(), np.array(scores['R']).mean(), np.array(scores['F1']).mean()
        print(f"BertScore scores:\n   Precision@{K}: {mP:.4f}\n   Recall@{K}: {mR:.4f}\n   F1@{K}: {mF1:.4f}")
        gc.collect()
        torch.cuda.empty_cache()
        return scores


class DilaougesScorer(DialogEvaluator):
    def evaluate(self, inputs):
        query, answer, context = inputs['query'], inputs['answer'], inputs['context']
        cands = [answer[0] for _ in range(len(context))]
        queries = [query[0] for _ in range(len(context))]
        K = len(context)

        scores = defaultdict(list)
        for i in range(K):
            # Prepare data for pre-trained evaluators

            data = convert_to_json(output_list=[cands[i]],
                                   src_list=[queries[i]], context_list=[context[i]])
            score = self.single_evaluate(data)
            [scores[key].append(item) for key, item in score[0].items()]

        print("\n\nUni Eval Sores")
        [print(f"   {key}@{K}: {np.array(value).mean():4f}") for key, value in scores.items()]
        scores['K'].append(K)
        gc.collect()
        torch.cuda.empty_cache()
        return scores


if __name__ == "__main__":
    scorer2 = DilaougesScorer()
    scores2 = scorer2.evaluate(Example)
    scorer1 = BertEvaluator(model_type='bert-base-uncased')
    # scorer1 = BertEvaluator(model_type='microsoft/deberta-xlarge-mnli')
    scores1 = scorer1.scores(Example)

    # print(scores1)
