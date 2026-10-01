import jenkins.model.Jenkins
import org.jenkinsci.plugins.workflow.job.WorkflowJob
import org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition
// Dedicated localhost-only demonstration controller. Never expose publicly.
def j = Jenkins.get()
j.setNumExecutors(1)
def job = j.getItem('heart-mlops') ?: j.createProject(WorkflowJob, 'heart-mlops')
job.setDefinition(new CpsFlowDefinition(new File('/submission/Jenkinsfile').text, false))
j.save()
