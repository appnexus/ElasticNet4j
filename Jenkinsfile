#!groovy
@Library('build-jenkins-pipelines')
import com.appnexus.ArtifactoryUtilities
import java.text.SimpleDateFormat

pipeline {
  agent {
    label 'optimization-worker-ubuntu'
  }

  environment {
    APP_NAME = 'tj-valuation'
    APP_REPO = 'app/tj-valuation'
    APP_ARTIFACTORY_DEPLOY_CREDENTIALS_ID = 'artifactory-deploy'
    APP_ARTIFACTORY_PROMOTE_CREDENTIALS_ID = '0dbbef00-aaab-4a64-8ee4-e057ef13d08e'
  }

  tools {
    maven 'Maven 3.3.9'
    jdk 'Oracle JDK 11.0.1'
  }

  options {
    buildDiscarder(
      logRotator(
        numToKeepStr: '30',
        daysToKeepStr: '14',
        artifactDaysToKeepStr: '7'
      )
    )
    timeout(time: 30, unit: 'MINUTES')
    disableConcurrentBuilds()
    timestamps()
  }

  stages {
    stage('init') {
      steps {
        sh 'mvn clean'
      }
    }

    stage('test') {
      steps {
        sh 'mvn test'
        junit keepLongStdio: true, testResults: 'target/surefire-reports/TEST-*.xml'
      }
    }

    stage('build') {
      steps {
        echo "building jar..."
        sh 'mvn package'
      }
    }

    stage('publish-release') {
      when {
        branch 'master'
      }

      environment {
        CZAR_AUTH = credentials('5b91284b-996f-4731-b592-8524d6fc4d28')
        MAESTRO_CREDENTIALS_ID = '5b91284b-996f-4731-b592-8524d6fc4d28'
        DEVOPS_JENKINS_SSH_CREDENTIALS_ID = 'f6a4ff83-ff98-416b-ba90-18b2c563de84'
        APP_VERSION =  sh(
                returnStdout: true,
                script: "ls target/${env.APP_NAME}-*.jar | grep -v shaded |  sed \"s/${env.APP_NAME}//g\" | cut -d '-' -f 2,3 | sed \"s/.jar//g\""
        ).trim()
        IS_SNAPSHOT = sh (
                returnStdout: true,
                script: "echo ${env.APP_VERSION} | grep -o '-' | wc -l"
        ).trim()
      }

      steps {

        script {
          sh "mvn clean deploy"
          echo "VERSION: ${env.APP_VERSION}"
          echo "IS_SNAPSHOT: ${env.IS_SNAPSHOT}"

          def dateFormat = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss")
          def date = new Date()
          fullVersion = env.APP_VERSION

          if (env.IS_SNAPSHOT.toInteger()) {
            // append timestamp to version if snapshot
            fullVersion = env.APP_VERSION + "-" + dateFormat.format(date)
            echo "Updated Snapshot Version: ${fullVersion}"
          } else {
            // tag if release version
            sh "git tag ${fullVersion}"
            sh "git push origin ${fullVersion}"
          }
        }
      }
    }
  }

  post {
    success {
      archive 'target/**/*.jar'
    }
    always {
      deleteDir()
    }
  }
}
